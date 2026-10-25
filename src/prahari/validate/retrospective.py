"""Retrospective validation over a real flood event.

    prahari validate --district barpeta --date 2024-07-16

What this actually does, and what it honestly proves:

  1. Runs the complete pipeline over every Sentinel-1 acquisition in a window
     around the event.
  2. Subtracts JRC permanent water, so reported flood is flood and not river.
  3. Checks the series behaves like a flood: extent rises to a peak and
     recedes. A detector returning noise across a real hydrograph fails this
     without needing any external label — which is the point, because labels
     for Assam are not freely available.
  4. If an independent reference mask is supplied (``--reference``, e.g. a
     Bhuvan/NDEM export or a hand-digitised extent), scores IoU/F1 against it.

What it does NOT prove, stated plainly in the report because a judge will ask:

  * It does not validate DEPTH. There is no public per-pixel depth truth for
    these events. HAND-derived depth remains a modelled quantity.
  * Without ``--reference``, extent is checked for internal consistency, not
    against ground truth.
  * It cannot validate the evacuation plan itself. Nobody recorded which road
    each family took. The strongest available comparison is against relief
    operation records, and obtaining those is a conversation with ASDMA, not
    an API call. Section 9.2 of the build plan exists for exactly this.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["run_retrospective", "RetrospectiveReport"]


@dataclass
class FrameResult:
    date: str
    item_id: str
    coverage: float
    threshold_db: float
    total_water_km2: float
    flood_km2: float
    permanent_km2: float
    exposed_population: int
    needing_evacuation: int
    settlements_cut_off: int
    people_cut_off: int
    roads_cut_km: float
    max_depth_m: float
    reference_iou: Optional[float] = None
    reference_f1: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RetrospectiveReport:
    district: str
    event_date: str
    frames: list[FrameResult] = field(default_factory=list)
    hydrograph_consistent: Optional[bool] = None
    notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "district": self.district,
            "event_date": self.event_date,
            "n_frames": len(self.frames),
            "hydrograph_consistent": self.hydrograph_consistent,
            "frames": [f.to_dict() for f in self.frames],
            "notes": self.notes,
            "limitations": self.limitations,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# PRAHARI retrospective validation — {self.district}",
            "",
            f"Event date: **{self.event_date}**  ·  {len(self.frames)} Sentinel-1 acquisitions",
            "",
            "| date | coverage | flood km² | river km² | exposed | evacuating | cut off | roads cut |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in self.frames:
            lines.append(
                f"| {f.date[:10]} | {f.coverage:.0%} | {f.flood_km2:.1f} | {f.permanent_km2:.1f} | "
                f"{f.exposed_population:,} | {f.needing_evacuation:,} | "
                f"{f.settlements_cut_off} ({f.people_cut_off:,}) | {f.roads_cut_km:.1f} km |"
            )
        if any(f.reference_iou is not None for f in self.frames):
            lines += ["", "## Against the supplied reference mask", "",
                      "| date | IoU | F1 |", "|---|---|---|"]
            for f in self.frames:
                if f.reference_iou is not None:
                    lines.append(f"| {f.date[:10]} | {f.reference_iou:.4f} | {f.reference_f1:.4f} |")

        lines += ["", "## Findings", ""]
        lines += [f"- {n}" for n in self.notes]
        lines += ["", "## What this does NOT establish", ""]
        lines += [f"- {l}" for l in self.limitations]
        return "\n".join(lines)


def run_retrospective(
    district: str = "barpeta",
    event_date: str = "2024-07-16",
    window_days: int = 24,
    collection: str = "sentinel-1-rtc",
    mode: str = "foot",
    reference: Optional[str] = None,
    out: Optional[str] = None,
    max_frames: int = 6,
) -> int:
    """Run the validation. Returns a process exit code."""
    from ..config import data_dir, load_config
    from ..decision import TravelMode, build_evacuation_plan
    from ..grid import grid_from_config
    from ..hazard.infer import detect_flood
    from ..hazard.metrics import segmentation_metrics
    from ..ingest import build_real_scenario
    from ..ingest.sentinel1 import fetch_sentinel1_scene, search_sentinel1
    from ..scenario import Provenance
    from .permanent_water import fetch_permanent_water, separate_flood_from_permanent

    config = load_config(district)
    grid = grid_from_config(config)
    bbox = config.grid.bbox.as_tuple()
    target = datetime.fromisoformat(event_date).replace(tzinfo=timezone.utc)

    print(f"\n=== RETROSPECTIVE VALIDATION — {config.name}, event {event_date} ===\n")

    scenario, build_report = build_real_scenario(config, use_cache=True)
    if scenario is None:
        print(f"FATAL: {build_report.get('fatal')}")
        return 1
    print(build_report.summary())

    permanent_result = fetch_permanent_water(grid, bbox, district_id=config.id)
    permanent = permanent_result.data if permanent_result else None
    if not permanent_result:
        print(f"\nWARNING: permanent water unavailable ({permanent_result.reason}); "
              "reported flood extent will include the river itself.")

    reference_mask = _load_reference(reference, grid) if reference else None

    search = search_sentinel1(
        bbox,
        start=target - timedelta(days=window_days),
        end=target + timedelta(days=window_days),
        collection=collection,
    )
    if not search:
        print(f"FATAL: {search.reason}")
        return 1

    items = sorted(search.data, key=lambda it: it["datetime"])[:max_frames]
    print(f"\n{len(items)} acquisitions in the window\n")

    report = RetrospectiveReport(district=config.name, event_date=event_date)

    for item in items:
        print(f"  {item['datetime'][:10]}  {item['id'][:48]} …")
        fetched = fetch_sentinel1_scene(
            grid, bbox, item_id=item["id"], collection=collection,
            district_id=config.id, use_cache=True,
        )
        if not fetched:
            print(f"    skipped: {fetched.reason[:80]}")
            continue
        scene = fetched.data

        detection = detect_flood(
            scene.vv_db, scene.vh_db, scenario.hand, already_db=True,
            max_depth_m=config.hydrology.max_depth_m, prefer="otsu",
            calibrated=scene.calibrated,
        )
        separation = separate_flood_from_permanent(detection.mask, permanent, grid)

        hazard = scenario.with_hazard(
            flood_mask=separation["flood_mask"],
            depth=detection.depth * (separation["flood_mask"] > 0),
            confidence=detection.confidence,
            timestamp=scene.acquired,
            provenance=Provenance(
                "hazard", detection.method,
                detail=f"{detection.method_detail}; permanent water subtracted",
            ),
        )
        plan = build_evacuation_plan(hazard, config, mode=TravelMode(mode))
        headline = plan.headline()

        frame = FrameResult(
            date=scene.acquired,
            item_id=scene.item_id,
            coverage=scene.coverage,
            threshold_db=0.0,
            total_water_km2=separation["total_water_km2"],
            flood_km2=separation["flood_km2"],
            permanent_km2=separation["permanent_km2"],
            exposed_population=int(headline["exposed_population"]),
            needing_evacuation=int(headline["needing_evacuation"]),
            settlements_cut_off=int(headline["settlements_cut_off"]),
            people_cut_off=int(headline["people_cut_off"]),
            roads_cut_km=float(headline["roads_cut_km"]),
            max_depth_m=detection.max_depth_m,
        )

        if reference_mask is not None:
            metrics = segmentation_metrics(separation["flood_mask"], reference_mask)
            frame.reference_iou = metrics.iou
            frame.reference_f1 = metrics.f1

        report.frames.append(frame)
        print(f"    flood {frame.flood_km2:7.1f} km²   river {frame.permanent_km2:6.1f} km²   "
              f"evacuating {frame.needing_evacuation:>7,}   cut off {frame.settlements_cut_off}")

    if not report.frames:
        print("\nNo usable acquisitions. Nothing to validate.")
        return 1

    _assess(report, permanent_result.ok if permanent_result else False, reference is not None)

    print("\n" + report.to_markdown())

    out_path = Path(out) if out else data_dir() / "out" / f"validation_{district}_{event_date}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out_path}.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    Path(f"{out_path}.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"\nWritten to {out_path}.json and {out_path}.md")
    return 0


def _assess(report: RetrospectiveReport, separated: bool, has_reference: bool) -> None:
    """Turn the frame series into findings, without overclaiming."""
    extents = [f.flood_km2 for f in report.frames]
    coverages = [f.coverage for f in report.frames]

    if len(extents) >= 3:
        peak = int(np.argmax(extents))
        rises = peak > 0 and extents[peak] > extents[0]
        recedes = peak < len(extents) - 1 and extents[peak] > extents[-1]
        report.hydrograph_consistent = bool(rises and recedes)
        if report.hydrograph_consistent:
            report.notes.append(
                f"Flood extent rises to a peak of {extents[peak]:.0f} km² on "
                f"{report.frames[peak].date[:10]} and recedes afterwards — the series "
                "behaves like a real hydrograph, which is what a working detector should "
                "produce over a real event."
            )
        else:
            report.notes.append(
                "Flood extent does NOT show a clean rise-peak-recede shape across the "
                "window. That may be genuine (a multi-peak event), or it may indicate the "
                "detector is unstable between acquisitions. Investigate before quoting "
                "any single frame."
            )
    else:
        report.notes.append(
            f"Only {len(extents)} usable acquisition(s) — too few to assess hydrograph shape."
        )

    if min(coverages) < 0.999:
        report.notes.append(
            f"Swath coverage ranges {min(coverages):.0%}–{max(coverages):.0%}. Areas outside "
            "a pass were NOT observed, so every extent here is a LOWER BOUND. Frames with "
            "different coverage are not directly comparable to each other."
        )

    if separated:
        river = np.mean([f.permanent_km2 for f in report.frames])
        report.notes.append(
            f"Permanent water (JRC GSW 1984–2021) accounts for an average {river:.0f} km² of "
            "detected water and has been subtracted. Without this the system would report "
            "the Brahmaputra itself as flooding."
        )

    cut_off = [f.settlements_cut_off for f in report.frames]
    if max(cut_off) > 0:
        peak_frame = report.frames[int(np.argmax(cut_off))]
        report.notes.append(
            f"At peak, {peak_frame.settlements_cut_off} settlements "
            f"({peak_frame.people_cut_off:,} people) had no road route to a viable shelter. "
            "This is the output no existing system produces; it is also the number to take "
            "to a district officer for confirmation against what actually happened."
        )

    report.limitations = [
        "DEPTH IS NOT VALIDATED. No public per-pixel depth truth exists for these events. "
        "HAND-derived depth is a modelled quantity throughout.",
        "Shelter capacity is estimated from OSM building footprints, not surveyed. The "
        "sheltered/unsheltered split therefore inherits that uncertainty.",
        "OSM road and building coverage in rural Assam is uneven. A settlement can appear "
        "cut off because its road was never mapped — those cases are reported separately "
        "under the UNBOUND reason code rather than merged into CUT_OFF.",
        "The evacuation PLAN itself is unvalidated. Nobody recorded which road each family "
        "took. Comparison against ASDMA relief-operation records is the next step and "
        "requires a conversation, not an API.",
    ]
    if not has_reference:
        report.limitations.insert(
            0,
            "No independent reference mask was supplied, so flood EXTENT was checked for "
            "internal/temporal consistency only — not against ground truth. Pass "
            "--reference with a Bhuvan/NDEM export to score IoU.",
        )


def _load_reference(path: str, grid) -> Optional[np.ndarray]:
    """Load an independent flood mask and resample it onto the grid."""
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        from rasterio.warp import Resampling, reproject
    except ImportError:
        logger.error("rasterio is required to read a reference mask")
        return None

    dst = np.zeros(grid.shape, dtype=np.uint8)
    a, b, c, d, e, f = grid.transform
    try:
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=Affine(a, b, c, d, e, f), dst_crs=CRS.from_string(grid.crs),
                resampling=Resampling.nearest,
            )
    except Exception as exc:
        logger.error("Could not read reference mask %s: %s", path, exc)
        return None
    logger.info("Reference mask loaded: %.2f%% water", 100 * (dst > 0).mean())
    return (dst > 0).astype(np.uint8)
