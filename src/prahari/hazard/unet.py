"""U-Net for SAR flood segmentation.

Architecture: U-Net with an optional pretrained ResNet-34 encoder. Two inputs
(Sentinel-1 VV + VH), one output (water logit).

Design notes worth defending out loud:

  * TWO CHANNELS, NOT THREE. Everyone reaches for a 3-channel ImageNet stem.
    SAR has two useful polarisations. We adapt the first conv by averaging the
    RGB weights across the channel dimension and re-tiling, which preserves
    the pretrained filter response instead of throwing it away or feeding a
    fake third channel.

  * dB, NOT LINEAR, INPUT. Backscatter is log-normal. Normalising in dB gives
    the network roughly Gaussian inputs; feeding linear sigma0 makes the loss
    surface far worse for no benefit.

  * DICE + BCE LOSS. Flood pixels are a small minority of most scenes. Pure
    BCE happily predicts "all dry" and scores well on accuracy. Dice attacks
    the overlap directly. We combine them.

  * TILED INFERENCE WITH OVERLAP. A district is far larger than the 256 px
    training tile. Naive tiling produces visible seams at tile borders, which
    on a flood map look exactly like embankments. We blend with a cosine
    window (see infer.py).

The model is optional. If torch is missing or no checkpoint exists, the
pipeline serves the Otsu baseline. Nothing downstream knows the difference.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Anaconda ships its own OpenMP runtime (libiomp5md.dll) and so does PyTorch.
# On Windows, importing both aborts the process with "OMP: Error #15" before a
# single tensor is allocated. Setting this before the torch import is the
# documented workaround; without it, training and inference simply refuse to
# start on a stock Anaconda install, which is the environment most teams have.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = object  # type: ignore

__all__ = ["UNet", "DiceBCELoss", "build_model", "TORCH_AVAILABLE"]


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """(conv -> BN -> ReLU) x 2, the standard U-Net double convolution."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class UpBlock(nn.Module):
        """Upsample, concatenate the skip connection, then convolve.

        Bilinear upsampling rather than a transposed convolution: transposed
        convs produce checkerboard artefacts, and on a flood boundary those
        artefacts read as real structure.
        """

        def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.conv = ConvBlock(out_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            return self.conv(torch.cat([x, skip], dim=1))

    class UNet(nn.Module):
        """U-Net; encoder is either a pretrained ResNet-34 or a plain CNN."""

        def __init__(
            self,
            in_channels: int = 2,
            n_classes: int = 1,
            base: int = 64,
            encoder: str = "resnet34",
            pretrained: bool = True,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.encoder_name = encoder

            if encoder.startswith("resnet"):
                self._build_resnet_encoder(encoder, in_channels, pretrained)
            else:
                self._build_plain_encoder(in_channels, base)

            self.head = nn.Conv2d(self.decoder_out, n_classes, 1)

        # -- encoders ------------------------------------------------------
        def _build_resnet_encoder(self, name: str, in_channels: int, pretrained: bool) -> None:
            try:
                import torchvision.models as tvm
            except ImportError:
                logger.warning("torchvision unavailable; falling back to the plain encoder")
                self._build_plain_encoder(in_channels, 64)
                return

            weights = "IMAGENET1K_V1" if pretrained else None
            try:
                net = getattr(tvm, name)(weights=weights)
            except Exception as exc:  # offline, or weights not cached
                logger.warning("Could not load pretrained %s (%s); training from scratch", name, exc)
                net = getattr(tvm, name)(weights=None)

            net.conv1 = _adapt_first_conv(net.conv1, in_channels)

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2  64ch
            self.pool = net.maxpool                                    # /4
            self.enc1, self.enc2 = net.layer1, net.layer2              # /4 64, /8 128
            self.enc3, self.enc4 = net.layer3, net.layer4              # /16 256, /32 512

            chans = (64, 64, 128, 256, 512)
            self.up4 = UpBlock(chans[4], chans[3], 256)
            self.up3 = UpBlock(256, chans[2], 128)
            self.up2 = UpBlock(128, chans[1], 64)
            self.up1 = UpBlock(64, chans[0], 32)
            self.decoder_out = 32
            self._mode = "resnet"

        def _build_plain_encoder(self, in_channels: int, base: int) -> None:
            self.inc = ConvBlock(in_channels, base)
            self.d1 = ConvBlock(base, base * 2)
            self.d2 = ConvBlock(base * 2, base * 4)
            self.d3 = ConvBlock(base * 4, base * 8)
            self.pool2 = nn.MaxPool2d(2)
            self.up3 = UpBlock(base * 8, base * 4, base * 4)
            self.up2 = UpBlock(base * 4, base * 2, base * 2)
            self.up1 = UpBlock(base * 2, base, base)
            self.decoder_out = base
            self._mode = "plain"

        # -- forward -------------------------------------------------------
        def forward(self, x):
            if self._mode == "resnet":
                s0 = self.stem(x)          # /2
                s1 = self.enc1(self.pool(s0))
                s2 = self.enc2(s1)
                s3 = self.enc3(s2)
                s4 = self.enc4(s3)
                y = self.up4(s4, s3)
                y = self.up3(y, s2)
                y = self.up2(y, s1)
                y = self.up1(y, s0)
                y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
            else:
                s0 = self.inc(x)
                s1 = self.d1(self.pool2(s0))
                s2 = self.d2(self.pool2(s1))
                s3 = self.d3(self.pool2(s2))
                y = self.up3(s3, s2)
                y = self.up2(y, s1)
                y = self.up1(y, s0)
            return self.head(y)

        @torch.no_grad()
        def predict_proba(self, x):
            return torch.sigmoid(self.forward(x))

    def _adapt_first_conv(conv: "nn.Conv2d", in_channels: int) -> "nn.Conv2d":
        """Retarget an RGB stem at ``in_channels`` while keeping pretrained weights.

        The RGB filters are averaged across the channel axis and tiled. The
        response magnitude is preserved (we rescale by 3/in_channels), so the
        network starts from something close to its pretrained behaviour rather
        than from noise in the first layer.
        """
        new = nn.Conv2d(
            in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, bias=conv.bias is not None,
        )
        with torch.no_grad():
            w = conv.weight.mean(dim=1, keepdim=True)          # (out, 1, k, k)
            new.weight.copy_(w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        return new

    class DiceBCELoss(nn.Module):
        """BCE + soft Dice, with support for ignored pixels.

        Sen1Floods11 marks unlabelled pixels as -1. Scoring them as either
        class trains the model on noise, so they are masked out of BOTH terms.
        """

        def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0,
                     pos_weight: Optional[float] = None) -> None:
            super().__init__()
            self.bce_weight = bce_weight
            self.smooth = smooth
            self.pos_weight = pos_weight

        def forward(self, logits, target, ignore_mask=None):
            if ignore_mask is None:
                valid = torch.ones_like(target)
            else:
                valid = (~ignore_mask.bool()).float()

            pw = (
                torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
                if self.pos_weight is not None else None
            )
            bce_map = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none", pos_weight=pw
            )
            denom = valid.sum().clamp(min=1.0)
            bce = (bce_map * valid).sum() / denom

            probs = torch.sigmoid(logits) * valid
            tgt = target.float() * valid
            inter = (probs * tgt).sum()
            dice = 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + tgt.sum() + self.smooth)

            return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


def build_model(
    in_channels: int = 2,
    encoder: str = "resnet34",
    pretrained: bool = True,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
):
    """Construct the model and optionally load a checkpoint.

    Raises if torch is absent — callers that must degrade gracefully should
    check :data:`TORCH_AVAILABLE` first (``infer.py`` does).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install the model extra "
            "(pip install 'prahari[model]') or run with the Otsu baseline."
        )
    model = UNet(in_channels=in_channels, encoder=encoder, pretrained=pretrained)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("Checkpoint mismatch — missing=%s unexpected=%s", missing, unexpected)
        logger.info("Loaded checkpoint %s", checkpoint)
    return model.to(device)
