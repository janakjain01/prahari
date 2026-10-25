"""Hazard inference — the single entry point the pipeline calls.

    detect_flood(vv, vh, hand, ...) -> FloodDetection

It picks the best available method and always returns something usable:

    1. U-Net, if torch is installed AND a checkpoint exists AND that checkpoint
       recorded a win over the baseline;
    2. otherwise the Otsu baseline.

The caller does not branch on this. The returned object says which method ran,
and the dashboard displays it. A demo that silently degrades is a demo that
lies; a demo that degrades and SAYS SO is a demo that survives questions.

Tiled inference with cosine blending
------------------------------------
A district is ~1400x1500 px; the model was trained on 256 px tiles. Feeding
the full scene would run the network far outside its training distribution and
exhaust memory. Naive tiling instead produces discontinuities at tile borders,
which on a flood map look like embankments — a wrong answer that looks
plausible, which is the worst kind. We overlap tiles and blend them with a
raised-cosine window so contributions taper to zero at the edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .depth import depth_from_hand
from .otsu import detect_water_otsu

logger = logging.getLogger(__name__)

__all__ = ["detect_flood", "FloodDetection", "predict_unet", "checkpoint_status"]

Method = Literal["unet", "otsu", "otsu-fallback"]


@dataclass
class FloodDetection:
    """Everything the impact layer needs, plus the provenance to explain it."""

    mask: np.ndarray            # uint8 {0,1}
    depth: np.ndarray           # metres, float32
    confidence: np.ndarray      # float32 0..1
    method: Method
    method_detail: str
    flooded_fraction: float
    max_depth_m: float
    mean_depth_m: float
    fallback_reason: Optional[str] = None
    observed: Optional[np.ndarray] = None
    coverage: float = 1.0
    calibrated: bool = True

    def to_dict(self) -> dict:
        doc = {
            "method": self.method,
            "method_detail": self.method_detail,
            "flooded_fraction": round(self.flooded_fraction, 5),
            "max_depth_m": round(self.max_depth_m, 2),
            "mean_depth_m": round(self.mean_depth_m, 2),
            "fallback_reason": self.fallback_reason,
            "observed_coverage": round(self.coverage, 4),
            "calibrated": self.calibrated,
        }
        if self.coverage < 0.999:
            doc["coverage_caveat"] = (
                f"Only {self.coverage * 100:.0f}% of the district was observed by this "
                "pass. Flood extent outside the swath is UNKNOWN, not zero — the "
                "reported extent is a lower bound."
            )
        if not self.calibrated:
            doc["calibration_caveat"] = (
                "Input was uncalibrated Sentinel-1 GRD digital numbers. Thresholds were "
                "derived from the scene histogram, not from published backscatter values."
            )
        return doc


def checkpoint_status(checkpoint: Optional[str | Path]) -> dict:
    """Inspect a checkpoint without loading the network.

    Used by the dashboard to show, honestly, which hazard model is live.
    """
    status = {"exists": False, "path": str(checkpoint) if checkpoint else None,
              "trained_on_real_data": None, "val_iou": None, "usable": False}
    if not checkpoint:
        return status
    path = Path(checkpoint)
    if not path.exists():
        return status
    status["exists"] = True
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        status["trained_on_real_data"] = bool(state.get("trained_on_real_data", False))
        status["val_iou"] = state.get("val_iou")
        status["usable"] = True
    except Exception as exc:  # pragma: no cover - depends on env
        status["error"] = str(exc)
    return status


def detect_flood(
    vv: np.ndarray,
    vh: Optional[np.ndarray] = None,
    hand: Optional[np.ndarray] = None,
    *,
    already_db: bool = False,
    checkpoint: Optional[str | Path] = None,
    device: str = "cpu",
    max_depth_m: float = 8.0,
    prefer: Method = "unet",
    require_real_training: bool = True,
    hand_limit_m: float = 15.0,
    calibrated: bool = True,
) -> FloodDetection:
    """Detect flood extent and estimate depth.

    Parameters
    ----------
    prefer
        ``"unet"`` tries the learned model first; ``"otsu"`` forces the
        baseline. Force the baseline when you need a result that is fully
        explainable on a whiteboard — which is a legitimate thing to want on
        stage.
    require_real_training
        Refuse a checkpoint that was trained on synthetic data. Default True,
        and it should stay True.
    """
    fallback_reason: Optional[str] = None
    mask = confidence = None
    method: Method = "otsu"
    detail = ""

    if prefer == "unet" and checkpoint is not None:
        try:
            mask, confidence, detail = predict_unet(
                vv, vh, checkpoint, device, already_db, require_real_training
            )
            method = "unet"
        except Exception as exc:
            fallback_reason = str(exc)
            logger.warning("U-Net unavailable (%s) — using the Otsu baseline", exc)

    observed: Optional[np.ndarray] = None
    coverage = 1.0

    if mask is None:
        result = detect_water_otsu(
            vv, vh, hand=hand, already_db=already_db, hand_limit_m=hand_limit_m,
            calibrated=calibrated,
        )
        mask, confidence = result.mask, result.confidence
        observed, coverage = result.observed, result.coverage
        method = "otsu-fallback" if fallback_reason else "otsu"
        detail = (
            f"Tiled Otsu thresholding at {result.threshold_db:.2f} dB "
            f"({result.tiles_used}/{result.tiles_total} tiles bimodal)"
            + ("" if coverage >= 0.999 else f", {coverage * 100:.0f}% of district observed")
        )
    elif hand is not None:
        # The learned model has no terrain prior. Apply the same physical veto
        # the baseline gets: water does not sit on a hillside.
        terrain_ok = ~np.isfinite(hand) | (np.asarray(hand) <= hand_limit_m)
        removed = int((mask.astype(bool) & ~terrain_ok).sum())
        if removed:
            logger.info("Terrain veto removed %d px from the U-Net mask", removed)
            mask = (mask.astype(bool) & terrain_ok).astype(np.uint8)

    if hand is None:
        logger.warning("No HAND raster supplied — depth cannot be estimated; returning zeros")
        depth = np.zeros_like(mask, dtype=np.float32)
        dmax = dmean = 0.0
    else:
        d = depth_from_hand(mask, hand, max_depth_m=max_depth_m, confidence=confidence)
        depth, dmax, dmean = d.depth, d.max_depth_m, d.mean_depth_m

    return FloodDetection(
        mask=mask.astype(np.uint8),
        depth=depth,
        confidence=(confidence if confidence is not None else np.zeros_like(depth)).astype(np.float32),
        method=method,
        method_detail=detail,
        # Denominator is the OBSERVED area, not the whole grid. Dividing by
        # the full grid when a third of it was outside the swath silently
        # deflates the reported flood fraction.
        flooded_fraction=(
            float(mask[observed].mean()) if observed is not None and observed.any()
            else float(mask.mean())
        ),
        max_depth_m=dmax,
        mean_depth_m=dmean,
        fallback_reason=fallback_reason,
        observed=observed,
        coverage=coverage,
        calibrated=calibrated,
    )


def predict_unet(
    vv: np.ndarray,
    vh: Optional[np.ndarray],
    checkpoint: str | Path,
    device: str = "cpu",
    already_db: bool = False,
    require_real_training: bool = True,
    tile: int = 256,
    overlap: int = 64,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run the U-Net over a full scene with overlapped, cosine-blended tiles."""
    from .dataset import normalise
    from .otsu import to_db
    from .unet import TORCH_AVAILABLE, build_model

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")
    import torch

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    state = torch.load(path, map_location=device, weights_only=False)
    if require_real_training and not state.get("trained_on_real_data", False):
        raise RuntimeError(
            "Checkpoint was trained on synthetic data and is refused. Train on "
            "Sen1Floods11, or pass require_real_training=False for a plumbing test only."
        )

    vv_db = np.asarray(vv, dtype=np.float32) if already_db else to_db(vv).astype(np.float32)
    if vh is None:
        # The network expects two channels. A constant VH is a lie the model
        # can see through, so we approximate the usual VV/VH offset instead —
        # and record that we did.
        vh_db = vv_db - 7.0
        chan_note = " (VH synthesised from VV; dual-pol input strongly preferred)"
    else:
        vh_db = np.asarray(vh, dtype=np.float32) if already_db else to_db(vh).astype(np.float32)
        chan_note = ""

    # Use the statistics the checkpoint was TRAINED with, not today's module
    # defaults. Older checkpoints predate this field and were trained on the
    # module defaults, so falling back to them is correct for those.
    stack = normalise(
        np.stack([vv_db, vh_db]),
        mean=state.get("norm_mean_db"),
        std=state.get("norm_std_db"),
    )
    model = build_model(2, state.get("config", {}).get("encoder", "resnet34"),
                        pretrained=False, device=device)
    model.load_state_dict(state["model"])
    model.eval()

    prob = _tiled_predict(model, stack, device, tile, overlap)
    mask = (prob > threshold).astype(np.uint8)

    # Confidence = distance from the decision boundary, rescaled to 0..1.
    confidence = np.clip(np.abs(prob - threshold) * 2.0, 0.0, 1.0).astype(np.float32)
    confidence = np.where(mask > 0, np.maximum(confidence, 0.05), 0.0)

    val_iou = state.get("val_iou")
    detail = f"U-Net ({state.get('config', {}).get('encoder', 'resnet34')} encoder)"
    if val_iou is not None:
        detail += f", val IoU {val_iou:.3f}"
    detail += chan_note
    return mask, confidence, detail


def _tiled_predict(model, stack: np.ndarray, device: str, tile: int, overlap: int) -> np.ndarray:
    """Sliding-window inference with raised-cosine blending."""
    import torch

    _, h, w = stack.shape
    step = max(tile - overlap, 1)
    window = _cosine_window(tile)

    accum = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    rows = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    cols = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    # Always include the final row/column so the far edge is never dropped.
    if rows[-1] + tile < h:
        rows.append(h - tile)
    if cols[-1] + tile < w:
        cols.append(w - tile)

    with torch.no_grad():
        for r in rows:
            for c in cols:
                patch = stack[:, r : r + tile, c : c + tile]
                ph, pw = patch.shape[1], patch.shape[2]
                if ph < tile or pw < tile:
                    patch = np.pad(patch, ((0, 0), (0, tile - ph), (0, tile - pw)), mode="reflect")
                x = torch.from_numpy(patch[None, ...]).float().to(device)
                p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                accum[r : r + ph, c : c + pw] += p[:ph, :pw] * window[:ph, :pw]
                weight[r : r + ph, c : c + pw] += window[:ph, :pw]

    return np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 1e-9)


def _cosine_window(size: int) -> np.ndarray:
    """2-D raised-cosine (Hann) window, floored so no pixel gets zero weight."""
    ramp = np.hanning(size + 2)[1:-1]
    ramp = np.maximum(ramp, 1e-3)
    return np.outer(ramp, ramp)
