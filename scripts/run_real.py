"""Run the full PRAHARI pipeline on REAL observations.

    python scripts/run_real.py --district barpeta --date 2024-07-16

This is the script that turns "we designed for real data" into "we ran on real
data". Everything it uses is free and keyless:

    Copernicus DEM GLO-30    Microsoft Planetary Computer
    Sentinel-1 RTC           Microsoft Planetary Computer
    OpenStreetMap            Overpass via osmnx
    WorldPop 2020            direct download

No IMD key, no Google Flood Hub key, no waitlist. That is the point.

The first run downloads and caches; later runs are fast. Cache everything
before demo day and the venue's network becomes irrelevant.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from prahari.config import load_config, data_dir
from prahari.decision import TravelMode, build_evacuation_plan
from prahari.delivery import build_cap_alert, validate_cap
from prahari.hazard.infer import detect_flood
from prahari.ingest import build_real_scenario
from prahari.ingest.sentinel1 import fetch_sentinel1_scene
from prahari.scenario import Provenance

logger = logging.getLogger("run_real")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRAHARI on real observations")
    parser.add_argument("--district", default="barpeta")
    parser.add_argument("--date", default="2024-07-16", help="target acquisition date (YYYY-MM-DD)")
    parser.add_argument("--collection", default="sentinel-1-rtc",
                        help="sentinel-1-rtc is calibrated; sentinel-1-grd is raw DN")
    parser.add_argument("--mode", default="foot", choices=["foot", "vehicle"])
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--save", default=None, help="write the scenario to this directory")
    parser.add_argument("--out", default=None, help="write the plan JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.district)
    target = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)

    print(f"\n=== PRAHARI — REAL DATA RUN — {config.name}, {config.district.state} ===\n")

    # ---- static layers -------------------------------------------------
    t0 = time.time()
    scenario, report = build_real_scenario(config, use_cache=args.use_cache)
    print(report.summary())
    if scenario is None:
        print(f"\nFATAL: {report.get('fatal')}")
        return 1
    print(f"\nStatic layers built in {time.time() - t0:.0f}s")

    # ---- hazard --------------------------------------------------------
    print(f"\nFetching Sentinel-1 near {target.date()} …")
    scene_result = fetch_sentinel1_scene(
        scenario.grid, config.grid.bbox.as_tuple(),
        target_date=target, collection=args.collection,
        district_id=config.id, use_cache=args.use_cache,
    )
    if not scene_result:
        print(f"FATAL: no SAR scene — {scene_result.reason}")
        return 1

    scene = scene_result.data
    print(f"  {scene.item_id}")
    print(f"  acquired {scene.acquired[:19]}  orbit {scene.orbit}")
    print(f"  coverage {scene.coverage:.0%}   calibrated: {scene.calibrated}")

    detection = detect_flood(
        scene.vv_db, scene.vh_db, scenario.hand,
        already_db=True,
        max_depth_m=config.hydrology.max_depth_m,
        prefer="otsu",
        calibrated=scene.calibrated,
    )
    print(f"\nHazard: {detection.method} — {detection.method_detail}")
    print(f"  flooded {detection.flooded_fraction:.1%} of OBSERVED area, "
          f"max depth {detection.max_depth_m:.2f} m")

    hazard_scenario = scenario.with_hazard(
        flood_mask=detection.mask,
        depth=detection.depth,
        confidence=detection.confidence,
        timestamp=scene.acquired,
        provenance=Provenance(
            layer="hazard",
            source=detection.method,
            detail=f"{detection.method_detail} | {scene_result.detail}",
            is_synthetic=False,
            is_estimate=True,
        ),
    )

    # ---- decision ------------------------------------------------------
    plan = build_evacuation_plan(hazard_scenario, config, mode=TravelMode(args.mode))

    print("\n=== HEADLINE ===")
    for key, value in plan.headline().items():
        print(f"  {key:32s} {value}")

    boat = plan.boat_dispatch_list(limit=10)
    print(f"\n=== BOAT DISPATCH ({len(plan.boat_dispatch_list())} settlements) ===")
    for entry in boat:
        print(f"  #{entry.rank:<3d} {entry.settlement_name:<24s} "
              f"{entry.people_at_risk:>7,} at risk   {entry.lat:.4f},{entry.lon:.4f}")
    if not boat:
        print("  (none — every settlement retains a road route)")

    alert = build_cap_alert(plan, config)
    problems = validate_cap(alert.xml)
    print(f"\nCAP 1.2: {alert.identifier}")
    print(f"  {'VALID' if not problems else 'INVALID: ' + '; '.join(problems)}")

    if plan.warnings:
        print("\n=== WARNINGS ===")
        for w in plan.warnings:
            print(f"  ! {w}")

    if args.save:
        path = hazard_scenario.save(args.save)
        print(f"\nScenario written to {path}")
    out = Path(args.out) if args.out else data_dir() / "out" / f"plan_{config.id}_{args.date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    print(f"Plan written to {out}")

    print(f"\nTotal wall time {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
