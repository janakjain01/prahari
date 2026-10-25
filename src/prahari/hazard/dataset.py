"""Sen1Floods11 dataset plumbing.

Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020) is a public, hand-labelled
flood segmentation dataset built from Sentinel-1 imagery over 11 real flood
events. It exists precisely for this task, which is why this project labels
nothing by hand.

Expected layout after download::

    <root>/v1.1/data/flood_events/HandLabeled/S1Hand/*.tif      2 bands: VV, VH (dB)
    <root>/v1.1/data/flood_events/HandLabeled/LabelHand/*.tif   -1 / 0 / 1
    <root>/v1.1/splits/flood_handlabeled/flood_{train,valid,test}_data.csv

Label semantics — getting this wrong silently poisons every metric:

    -1  not annotated  -> IGNORE in loss and in evaluation
     0  not water
     1  water

Download (no auth required for the public bucket)::

    gsutil -m cp -r gs://sen1floods11/v1.1 <root>/

If the dataset is absent, :class:`SyntheticFloodDataset` provides a
structurally identical stand-in so the training code stays runnable and
testable. It is clearly marked synthetic and must never be used to produce a
number that appears on a slide.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Sen1Floods11Dataset",
    "SyntheticFloodDataset",
    "build_datasets",
    "NORM_MEAN_DB",
    "NORM_STD_DB",
]

Split = Literal["train", "valid", "test"]

# Per-channel normalisation in dB. VH is both darker and more variable than
# VV, so a single shared statistic would leave the second channel badly scaled.
#
# THESE ARE THE LEGACY VALUES and they are NOT the true Sen1Floods11 statistics
# — they were estimated rather than measured. Measured over the hand-labelled
# split the real figures are:
#
#     VV  mean -9.69  std 3.43        (legacy assumed -12.0 / 5.0)
#     VH  mean -16.09 std 3.94        (legacy assumed -19.0 / 5.5)
#
# They are kept as the DEFAULT because normalisation only has to be
# *consistent* between training and inference, not correct in absolute terms —
# a fixed offset is something the network simply learns around. Changing these
# constants would silently invalidate every checkpoint trained with them.
#
# New checkpoints therefore carry their own statistics (see
# MEASURED_SEN1FLOODS11_* and train.py), and infer.py prefers whatever the
# checkpoint stored. That way the constants can be corrected without breaking
# anything already trained.
NORM_MEAN_DB = np.array([-12.0, -19.0], dtype=np.float32)
NORM_STD_DB = np.array([5.0, 5.5], dtype=np.float32)

# Measured over the Sen1Floods11 hand-labelled split. Use these for new runs.
MEASURED_SEN1FLOODS11_MEAN_DB = np.array([-9.69, -16.09], dtype=np.float32)
MEASURED_SEN1FLOODS11_STD_DB = np.array([3.43, 3.94], dtype=np.float32)

IGNORE_LABEL = -1


@dataclass
class Sample:
    image: np.ndarray      # (2, H, W) float32, normalised
    label: np.ndarray      # (1, H, W) float32 in {0, 1}
    ignore: np.ndarray     # (1, H, W) bool — True where unannotated
    name: str


def normalise(
    db: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Standardise dB backscatter per channel.

    ``mean``/``std`` override the module defaults so inference can use exactly
    the statistics its checkpoint was trained with. Mismatching them shifts
    every input the network sees and quietly degrades predictions with no
    error anywhere.
    """
    arr = np.asarray(db, dtype=np.float32)
    m = NORM_MEAN_DB if mean is None else np.asarray(mean, dtype=np.float32)
    s = NORM_STD_DB if std is None else np.asarray(std, dtype=np.float32)
    return (arr - m.reshape(-1, 1, 1)[: arr.shape[0]]) / s.reshape(-1, 1, 1)[: arr.shape[0]]


class _BaseDataset:
    """Minimal sequence protocol; also works as a torch Dataset via duck typing."""

    def __len__(self) -> int:  # pragma: no cover - trivial
        raise NotImplementedError

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]


class Sen1Floods11Dataset(_BaseDataset):
    """Hand-labelled split of Sen1Floods11."""

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        tile_size: int = 256,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.tile_size = tile_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.image_dir = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
        self.label_dir = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
        self.pairs = self._load_split()

        if not self.pairs:
            raise FileNotFoundError(
                f"No Sen1Floods11 samples found for split '{split}' under {self.root}. "
                "Download with: gsutil -m cp -r gs://sen1floods11/v1.1 <root>/"
            )
        logger.info("Sen1Floods11[%s]: %d samples", split, len(self.pairs))

    def _load_split(self) -> list[tuple[Path, Path]]:
        split_csv = (
            self.root / "v1.1" / "splits" / "flood_handlabeled" / f"flood_{self.split}_data.csv"
        )
        pairs: list[tuple[Path, Path]] = []

        if split_csv.exists():
            with split_csv.open(newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if not row:
                        continue
                    img = self.image_dir / Path(row[0].strip()).name
                    lbl = (
                        self.label_dir / Path(row[1].strip()).name
                        if len(row) > 1
                        else self.label_dir / img.name.replace("_S1Hand", "_LabelHand")
                    )
                    if img.exists() and lbl.exists():
                        pairs.append((img, lbl))
        else:
            # No split file: fall back to pairing by filename and slicing
            # deterministically, so results stay reproducible.
            logger.warning("Split CSV missing at %s; deriving a deterministic split", split_csv)
            images = sorted(self.image_dir.glob("*.tif"))
            for img in images:
                lbl = self.label_dir / img.name.replace("_S1Hand", "_LabelHand")
                if lbl.exists():
                    pairs.append((img, lbl))
            n = len(pairs)
            cuts = {"train": slice(0, int(0.7 * n)),
                    "valid": slice(int(0.7 * n), int(0.85 * n)),
                    "test": slice(int(0.85 * n), n)}
            pairs = pairs[cuts[self.split]]
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Sample:
        img_path, lbl_path = self.pairs[idx]
        image = _read_raster(img_path)                 # (2, H, W) dB
        label = _read_raster(lbl_path)[0]              # (H, W) in {-1, 0, 1}

        image = np.nan_to_num(image, nan=-25.0, posinf=0.0, neginf=-40.0)
        ignore = label == IGNORE_LABEL
        binary = (label == 1).astype(np.float32)

        image, binary, ignore = self._crop(image, binary, ignore)
        if self.augment:
            image, binary, ignore = self._augment(image, binary, ignore)

        return Sample(
            image=normalise(image),
            label=binary[None, ...],
            ignore=ignore[None, ...],
            name=img_path.stem,
        )

    def _crop(self, image, label, ignore):
        """Random crop when training, deterministic centre crop otherwise.

        Evaluation must not move between runs, or the held-out number that
        justifies shipping the model is not reproducible.
        """
        t = self.tile_size
        _, h, w = image.shape
        if h < t or w < t:
            pad_h, pad_w = max(0, t - h), max(0, t - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-25.0)
            label = np.pad(label, ((0, pad_h), (0, pad_w)))
            ignore = np.pad(ignore, ((0, pad_h), (0, pad_w)), constant_values=True)
            _, h, w = image.shape
        if h == t and w == t:
            return image, label, ignore
        if self.split == "train":
            r = int(self.rng.integers(0, h - t + 1))
            c = int(self.rng.integers(0, w - t + 1))
        else:
            r, c = (h - t) // 2, (w - t) // 2
        return image[:, r : r + t, c : c + t], label[r : r + t, c : c + t], ignore[r : r + t, c : c + t]

    def _augment(self, image, label, ignore):
        """Flips and 90-degree rotations only.

        Deliberately no photometric jitter: SAR backscatter values are
        physically meaningful (a dB value IS the surface response), so
        randomly brightening them teaches the network that water can be bright.
        Geometric augmentation is safe; radiometric augmentation is not.
        """
        if self.rng.random() < 0.5:
            image, label, ignore = image[:, :, ::-1], label[:, ::-1], ignore[:, ::-1]
        if self.rng.random() < 0.5:
            image, label, ignore = image[:, ::-1, :], label[::-1, :], ignore[::-1, :]
        k = int(self.rng.integers(0, 4))
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k)
            ignore = np.rot90(ignore, k)
        return np.ascontiguousarray(image), np.ascontiguousarray(label), np.ascontiguousarray(ignore)


class SyntheticFloodDataset(_BaseDataset):
    """Structurally identical stand-in used when Sen1Floods11 is not present.

    It keeps the training code exercisable and the test suite green offline.
    It is NOT a substitute for the real dataset, and
    :func:`prahari.hazard.train.train` refuses to write a shippable checkpoint
    from it without an explicit override.
    """

    def __init__(self, n: int = 64, tile_size: int = 256, seed: int = 0, augment: bool = False) -> None:
        self.n = n
        self.tile_size = tile_size
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.is_synthetic = True

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Sample:
        rng = np.random.default_rng(idx * 7919 + 13)
        t = self.tile_size
        yy, xx = np.mgrid[0:t, 0:t].astype(np.float32)

        # A meandering channel that spills into irregular floodplain patches.
        phase, amp, width = rng.uniform(0, 6.28), rng.uniform(10, 45), rng.uniform(6, 26)
        centre = t / 2 + amp * np.sin(xx / rng.uniform(20, 60) + phase)
        water = np.abs(yy - centre) < width

        for _ in range(int(rng.integers(1, 5))):
            cy, cx = rng.uniform(0, t), rng.uniform(0, t)
            rad = rng.uniform(8, 38)
            blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < rad ** 2
            water |= blob & (rng.random() > 0.35)

        vv = np.where(water, -19.5, -8.5) + rng.normal(0, 1.8, (t, t))
        vh = np.where(water, -26.0, -15.0) + rng.normal(0, 2.0, (t, t))
        image = np.stack([vv, vh]).astype(np.float32)

        ignore = rng.random((t, t)) < 0.02   # mimic Sen1Floods11's unlabelled pixels
        return Sample(
            image=normalise(image),
            label=water.astype(np.float32)[None, ...],
            ignore=ignore[None, ...],
            name=f"synthetic_{idx:05d}",
        )


def build_datasets(
    root: Optional[str | Path],
    tile_size: int = 256,
    seed: int = 0,
) -> tuple[_BaseDataset, _BaseDataset, _BaseDataset, bool]:
    """Return ``(train, valid, test, is_real)``.

    Falls back to synthetic data with a loud warning rather than crashing, so
    the pipeline can be developed and tested before the download finishes.
    """
    if root is not None and Path(root).exists():
        try:
            return (
                Sen1Floods11Dataset(root, "train", tile_size, augment=True, seed=seed),
                Sen1Floods11Dataset(root, "valid", tile_size, augment=False, seed=seed),
                Sen1Floods11Dataset(root, "test", tile_size, augment=False, seed=seed),
                True,
            )
        except FileNotFoundError as exc:
            logger.warning("Sen1Floods11 present but unusable (%s); using synthetic data", exc)

    logger.warning(
        "SYNTHETIC DATA IN USE. Sen1Floods11 was not found at %s. The pipeline is "
        "exercisable but any metric produced from this is NOT a real result and must "
        "not appear on a slide.", root,
    )
    return (
        SyntheticFloodDataset(256, tile_size, seed, augment=True),
        SyntheticFloodDataset(48, tile_size, seed + 1),
        SyntheticFloodDataset(48, tile_size, seed + 2),
        False,
    )


def _read_raster(path: Path) -> np.ndarray:
    """Read a GeoTIFF as ``(bands, H, W)`` float32."""
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "rasterio is required to read Sen1Floods11 GeoTIFFs. "
            "Install the geo extra: pip install 'prahari[geo]'"
        ) from exc
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)
