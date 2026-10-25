"""Segmentation metrics — the numbers that decide whether the U-Net ships.

Rule for this project: the learned model goes into the demo only if it beats
the Otsu baseline on a held-out set, and we show both numbers on the slide. If
it loses, we ship the baseline and say so. A team that can show a model losing
to a baseline and choosing the baseline reads as more competent, not less.

IoU is the headline because it is the standard for flood segmentation and is
unforgiving about both over- and under-prediction. Recall is reported next to
it because in an evacuation the two errors are NOT symmetric: missing flooded
ground is far worse than flagging dry ground as wet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

__all__ = ["SegmentationMetrics", "segmentation_metrics", "compare_to_baseline"]


@dataclass
class SegmentationMetrics:
    iou: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    n_evaluated: int

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"IoU={self.iou:.4f} F1={self.f1:.4f} "
            f"P={self.precision:.4f} R={self.recall:.4f} Acc={self.accuracy:.4f}"
        )


def segmentation_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    ignore_mask: Optional[np.ndarray] = None,
) -> SegmentationMetrics:
    """Binary segmentation metrics for water vs not-water.

    ``ignore_mask`` excludes pixels from the count entirely. Sen1Floods11
    labels use -1 for "not annotated", and scoring those as either class would
    quietly corrupt every number reported.
    """
    pred = np.asarray(prediction) > 0
    true = np.asarray(truth) > 0

    if ignore_mask is not None:
        valid = ~np.asarray(ignore_mask, dtype=bool)
        pred = pred[valid]
        true = true[valid]

    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))
    n = tp + fp + fn + tn

    union = tp + fp + fn
    iou = tp / union if union else 1.0                    # both empty => perfect
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0

    return SegmentationMetrics(
        iou=iou, f1=f1, precision=precision, recall=recall, accuracy=accuracy,
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
        n_evaluated=n,
    )


def compare_to_baseline(
    model: SegmentationMetrics,
    baseline: SegmentationMetrics,
    min_iou_gain: float = 0.02,
) -> dict:
    """Decide whether the learned model has earned its place.

    ``min_iou_gain`` guards against shipping a model that is ahead by noise. A
    2-point IoU gain on a held-out set is a real improvement; 0.3 points is a
    different random seed.
    """
    gain = model.iou - baseline.iou
    ships = gain >= min_iou_gain
    return {
        "model": model.to_dict(),
        "baseline": baseline.to_dict(),
        "iou_gain": gain,
        "f1_gain": model.f1 - baseline.f1,
        "recall_gain": model.recall - baseline.recall,
        "min_iou_gain_required": min_iou_gain,
        "model_ships": ships,
        "verdict": (
            f"U-Net ships: +{gain:.4f} IoU over the Otsu baseline."
            if ships
            else (
                f"U-Net does NOT ship: {gain:+.4f} IoU over baseline is below the "
                f"{min_iou_gain:.2f} threshold. Serving the classical baseline instead — "
                "it is the honest choice and we say so on the slide."
            )
        ),
    }
