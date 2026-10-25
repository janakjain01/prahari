"""Hazard layer — where is the water, and how deep.

Two independent paths, on purpose:

    otsu.py   classical SAR water detection. No training, no GPU, no
              checkpoint. Always works. This is the BASELINE, and it is built
              first so the learned model has something to beat.

    unet.py   U-Net segmentation trained on Sen1Floods11. Better, but it is
              only allowed into the demo if it beats the baseline on a
              held-out set, with the numbers shown.

Building the baseline first is not caution, it is the argument: it proves the
ML earns its place instead of being decoration. A panel with a real ML person
on it will ask exactly this.

Depth comes from HAND (hand.py) — established hydrology, not a neural network,
because a neural network would be worse here and we should say so.
"""

from .hand import compute_hand, fill_depressions, flow_accumulation, d8_receivers
from .otsu import otsu_threshold, detect_water_otsu
from .depth import depth_from_hand, DepthResult
from .metrics import segmentation_metrics, SegmentationMetrics

__all__ = [
    "compute_hand",
    "fill_depressions",
    "flow_accumulation",
    "d8_receivers",
    "otsu_threshold",
    "detect_water_otsu",
    "depth_from_hand",
    "DepthResult",
    "segmentation_metrics",
    "SegmentationMetrics",
]
