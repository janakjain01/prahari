"""Training loop for the SAR flood segmentation U-Net.

Run on Kaggle Notebooks (~30 GPU-hours/week free) or Colab. Sen1Floods11's
hand-labelled split is small enough that this is comfortable — do not train
locally on CPU.

    python -m prahari.hazard.train --data /kaggle/input/sen1floods11 --epochs 40

What this script does that a stock training loop does not, and why each item
is worth a sentence to a panel:

  * It evaluates the OTSU BASELINE on the same held-out test set, in the same
    run, and writes both numbers into one metrics file. The claim "our model
    beats the classical method" then has a single reproducible artefact behind
    it instead of two runs someone has to trust were comparable.

  * It refuses to mark a checkpoint shippable if it trained on synthetic data.
    The guard exists because the most embarrassing possible outcome is quoting
    a validation number that came from data we generated ourselves.

  * It selects on validation IoU, not on loss. Loss improvements that do not
    move IoU are not improvements for this task.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset import build_datasets
from .metrics import compare_to_baseline, segmentation_metrics
from .otsu import detect_water_otsu
from .unet import TORCH_AVAILABLE, DiceBCELoss, build_model

logger = logging.getLogger(__name__)

__all__ = ["train", "TrainConfig", "evaluate"]


@dataclass
class TrainConfig:
    data_root: Optional[str] = None
    out_dir: str = "models"
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    tile_size: int = 256
    encoder: str = "resnet34"
    pretrained: bool = True
    num_workers: int = 2
    seed: int = 42
    # Sen1Floods11's hand-labelled validation split is only ~89 tiles, so val
    # IoU is NOISY between epochs — it swung 0.61 -> 0.49 -> 0.61 on
    # consecutive epochs in practice. A short patience reads that noise as a
    # plateau and stops while the model is still learning: the first run here
    # halted at epoch 10 with its best at epoch 2, having completed only a
    # quarter of a cosine schedule built for 30. Patience must exceed the
    # metric's natural oscillation period, not just "a few epochs".
    patience: int = 25
    device: Optional[str] = None
    allow_synthetic_checkpoint: bool = False


def _collate(batch):
    import torch

    images = torch.from_numpy(np.stack([b.image for b in batch]))
    labels = torch.from_numpy(np.stack([b.label for b in batch]))
    ignores = torch.from_numpy(np.stack([b.ignore for b in batch]))
    return images, labels, ignores


def train(cfg: TrainConfig) -> dict:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for training: pip install 'prahari[model]'")

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)
    if device == "cpu":
        logger.warning(
            "Training on CPU. This is fine for a smoke test and far too slow for a "
            "real run — use Kaggle or Colab for the checkpoint you actually ship."
        )

    train_ds, valid_ds, test_ds, is_real = build_datasets(cfg.data_root, cfg.tile_size, cfg.seed)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=_collate, drop_last=True,
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_collate,
    )

    model = build_model(2, cfg.encoder, cfg.pretrained, device=device)

    # Water is the minority class in most scenes. Weighting the positive class
    # up counteracts the "predict all dry" attractor that pure BCE falls into.
    criterion = DiceBCELoss(bce_weight=0.5, pos_weight=2.0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)

    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "unet_sar_flood.pt"

    best_iou, best_epoch, history = -1.0, -1, []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for images, labels, ignores in train_dl:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ignores = ignores.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels, ignores)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            running += float(loss.item())

        scheduler.step()
        train_loss = running / max(len(train_dl), 1)
        val = evaluate(model, valid_dl, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val.to_dict()})
        logger.info(
            "epoch %2d/%d  loss %.4f  val %s  lr %.2e",
            epoch + 1, cfg.epochs, train_loss, val, scheduler.get_last_lr()[0],
        )

        if val.iou > best_iou:
            best_iou, best_epoch = val.iou, epoch
            from .dataset import NORM_MEAN_DB, NORM_STD_DB

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_iou": val.iou,
                    "trained_on_real_data": is_real,
                    # Ship the normalisation WITH the weights. Inference that
                    # standardises differently from training silently degrades
                    # every prediction and raises no error anywhere.
                    "norm_mean_db": NORM_MEAN_DB.tolist(),
                    "norm_std_db": NORM_STD_DB.tolist(),
                },
                ckpt_path,
            )
            logger.info("  -> new best (IoU %.4f), checkpoint saved", val.iou)
        elif epoch - best_epoch >= cfg.patience:
            logger.info("Early stopping: no val IoU improvement in %d epochs", cfg.patience)
            break

    # -- final head-to-head on the untouched test split ---------------------
    test_dl = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_collate,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model_metrics = evaluate(model, test_dl, device)
    baseline_metrics = evaluate_otsu_baseline(test_ds)
    comparison = compare_to_baseline(model_metrics, baseline_metrics)

    report = {
        "trained_on_real_data": is_real,
        "dataset": "Sen1Floods11 (hand-labelled)" if is_real else "SYNTHETIC — not a real result",
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "elapsed_seconds": round(time.time() - t0, 1),
        "device": device,
        "config": asdict(cfg),
        "history": history,
        "test": comparison,
    }

    if not is_real and not cfg.allow_synthetic_checkpoint:
        report["WARNING"] = (
            "Trained on SYNTHETIC data. These metrics are meaningless as evidence and "
            "must not be presented. Download Sen1Floods11 and retrain before the demo."
        )
        logger.error(report["WARNING"])

    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("%s", comparison["verdict"])
    return report


def evaluate(model, dataloader, device: str):
    """Validation/test pass, accumulating a single global confusion matrix.

    Global rather than per-tile-averaged: averaging IoU over tiles lets a tile
    containing three water pixels count as much as one that is half flooded,
    which flatters the number. The aggregate is the honest statistic.
    """
    import torch

    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for images, labels, ignores in dataloader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            pred = probs > 0.5
            true = labels.numpy() > 0.5
            valid = ~ignores.numpy().astype(bool)

            tp += int(np.sum(pred & true & valid))
            fp += int(np.sum(pred & ~true & valid))
            fn += int(np.sum(~pred & true & valid))
            tn += int(np.sum(~pred & ~true & valid))

    return _metrics_from_counts(tp, fp, fn, tn)


def evaluate_otsu_baseline(dataset):
    """Run the classical baseline over the same test tiles.

    The tiles are normalised dB, so they are de-normalised back to dB before
    thresholding — Otsu operates on physical backscatter, not on z-scores.
    """
    from .dataset import NORM_MEAN_DB, NORM_STD_DB

    tp = fp = fn = tn = 0
    for sample in dataset:
        vv_db = sample.image[0] * NORM_STD_DB[0] + NORM_MEAN_DB[0]
        vh_db = sample.image[1] * NORM_STD_DB[1] + NORM_MEAN_DB[1]
        result = detect_water_otsu(
            vv_db, vh_db, already_db=True, tile_size=128, min_blob_pixels=8
        )
        pred = result.mask > 0
        true = sample.label[0] > 0.5
        valid = ~sample.ignore[0].astype(bool)

        tp += int(np.sum(pred & true & valid))
        fp += int(np.sum(pred & ~true & valid))
        fn += int(np.sum(~pred & true & valid))
        tn += int(np.sum(~pred & ~true & valid))

    return _metrics_from_counts(tp, fp, fn, tn)


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int):
    from .metrics import SegmentationMetrics

    union = tp + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    n = tp + fp + fn + tn
    return SegmentationMetrics(
        iou=tp / union if union else 1.0,
        f1=2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        precision=precision,
        recall=recall,
        accuracy=(tp + tn) / n if n else 0.0,
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
        n_evaluated=n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PRAHARI SAR flood segmentation model")
    parser.add_argument("--data", dest="data_root", default=None, help="Sen1Floods11 root")
    parser.add_argument("--out", dest="out_dir", default="models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(TrainConfig(**vars(args)))


if __name__ == "__main__":  # pragma: no cover
    main()
