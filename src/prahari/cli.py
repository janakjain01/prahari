"""Command-line interface.

    prahari serve                 run the dashboard
    prahari demo                  synthetic end-to-end run, no network
    prahari real                  run on real observations
    prahari validate              retrospective validation against an event
    prahari ingest                pre-fetch and cache everything
    prahari fetch-population      one-time WorldPop country download
    prahari cap                   emit a CAP 1.2 alert
    prahari status                what is configured and what is not

``ingest`` is the one to run the night before a demo. It fills the on-disk
cache so that nothing on stage depends on the venue's network.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("prahari")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("rasterio").setLevel(logging.WARNING)


# --------------------------------------------------------------------------
def cmd_serve(args) -> int:
    import os

    import uvicorn

    os.environ["PRAHARI_DOWNSAMPLE"] = str(args.downsample)
    print(f"\n  PRAHARI dashboard  →  http://{args.host}:{args.port}\n")
    uvicorn.run("prahari.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_demo(args) -> int:
    from .config import load_config
    from .decision import TravelMode, build_evacuation_plan
    from .delivery import build_cap_alert, validate_cap
    from .demo.synth import SynthConfig, synthesise_district, synthesise_sar
    from .hazard.infer import detect_flood
    from .hazard.metrics import segmentation_metrics
    from .scenario import Provenance

    config = load_config(args.district)
    print(f"\n=== PRAHARI SYNTHETIC DEMO — {config.name} ===")
    print("    Everything below is generated. No real observation is involved.\n")

    scenario = synthesise_district(config, SynthConfig(downsample=args.downsample))
    vv, vh, truth = synthesise_sar(scenario, stage_m=args.stage, seed=args.seed)

    detection = detect_flood(
        vv, vh, scenario.hand, already_db=True,
        max_depth_m=config.hydrology.max_depth_m, prefer="otsu",
    )
    metrics = segmentation_metrics(detection.mask, truth)
    print(f"Hazard detector vs known ground truth: {metrics}")

    hazard = scenario.with_hazard(
        detection.mask, detection.depth, detection.confidence,
        provenance=Provenance("hazard", detection.method,
                              detail=detection.method_detail, is_synthetic=True),
    )
    plan = build_evacuation_plan(hazard, config, mode=TravelMode(args.mode))

    print("\n=== HEADLINE ===")
    for key, value in plan.headline().items():
        print(f"  {key:32s} {value}")

    boat = plan.boat_dispatch_list(limit=8)
    print(f"\n=== BOAT DISPATCH ({len(plan.boat_dispatch_list())}) ===")
    for e in boat:
        print(f"  #{e.rank:<3d} {e.settlement_name:<26s} {e.people_at_risk:>7,} at risk")
    if not boat:
        print("  (none at this water level)")

    steered = [r for r in plan.routes if r.naive_route_would_flood][:5]
    if steered:
        print("\n=== ROUTES STEERED AWAY FROM FLOODED ROADS ===")
        for r in steered:
            print(f"  {r.settlement_name} → {r.shelter_name}")
            print(f"     avoids {', '.join(r.avoided_roads[:2])} "
                  f"(up to {r.naive_route_max_depth_m:.2f} m of water)")

    alert = build_cap_alert(plan, config)
    problems = validate_cap(alert.xml)
    print(f"\nCAP 1.2 {alert.identifier}: {'VALID' if not problems else problems}")

    if args.out:
        Path(args.out).write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        print(f"\nPlan written to {args.out}")
    return 0


def cmd_real(args) -> int:
    sys.argv = ["run_real.py", "--district", args.district, "--date", args.date,
                "--mode", args.mode, "--collection", args.collection]
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_real.py"
    if not script.exists():
        print(f"scripts/run_real.py not found at {script}", file=sys.stderr)
        return 1
    code = compile(script.read_text(encoding="utf-8"), str(script), "exec")
    namespace = {"__name__": "__main__", "__file__": str(script)}
    try:
        exec(code, namespace)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def cmd_ingest(args) -> int:
    """Pre-fetch and cache everything. Run this the night before a demo."""
    from datetime import timedelta

    from .config import load_config
    from .grid import grid_from_config
    from .ingest import build_real_scenario
    from .ingest.sentinel1 import fetch_sentinel1_scene, search_sentinel1
    from .ingest.weather import fetch_precipitation

    config = load_config(args.district)
    grid = grid_from_config(config)
    bbox = config.grid.bbox.as_tuple()
    lon, lat = config.grid.bbox.centroid

    print(f"\n=== PRE-FETCHING {config.name} ===")
    print("Caching to disk so demo day does not depend on the venue network.\n")

    scenario, report = build_real_scenario(config, use_cache=True,
                                           allow_population_download=args.population)
    print(report.summary())

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    found = search_sentinel1(bbox, start=start, end=end, collection=args.collection)
    print(f"\nSentinel-1 search: {'ok' if found else found.reason}")
    if found:
        for item in found.data[: args.scenes]:
            print(f"  caching {item['id'][:52]} ({item['datetime'][:10]}) …")
            result = fetch_sentinel1_scene(
                grid, bbox, item_id=item["id"], collection=args.collection,
                district_id=config.id, use_cache=True,
            )
            status = "cached" if result else f"FAILED: {result.reason[:70]}"
            print(f"    {status}")

    precip = fetch_precipitation(lon, lat)
    print(f"\nPrecipitation: {'ok' if precip else precip.reason[:80]}")
    print("\nDone. Re-run with --population to also pull the WorldPop country raster.")
    return 0


def cmd_fetch_population(args) -> int:
    from .config import load_config
    from .grid import grid_from_config
    from .ingest.population import fetch_worldpop

    config = load_config(args.district)
    grid = grid_from_config(config)
    print("\nThis performs a LARGE one-time download (~490 MB for India).")
    result = fetch_worldpop(
        grid, config.grid.bbox.as_tuple(), district_id=config.id,
        allow_full_download=True, use_cache=True,
    )
    print(f"\n{'OK: ' + result.detail if result else 'FAILED: ' + result.reason}")
    return 0 if result else 1


def cmd_fetch_training_data(args) -> int:
    from .ingest.sen1floods11 import download_sen1floods11

    print("\nSen1Floods11 is a PUBLIC bucket — no account, no key, no registration.")
    print("Hand-labelled subset: 446 image/label pairs, ~732 MB.\n")
    report = download_sen1floods11(args.root)
    print(json.dumps(report, indent=2))
    if report["failed"] == 0:
        print(f"\nNow train with:\n  python -m prahari.cli train --data {args.root}")
    return 0 if report["failed"] == 0 else 1


def cmd_train(args) -> int:
    from .hazard.train import TrainConfig, train

    report = train(
        TrainConfig(
            data_root=args.data, out_dir=args.out, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, tile_size=args.tile_size,
            encoder=args.encoder, num_workers=args.workers, seed=args.seed,
        )
    )
    print("\n" + report["test"]["verdict"])
    return 0


def cmd_validate(args) -> int:
    from .validate.retrospective import run_retrospective

    return run_retrospective(
        district=args.district, event_date=args.date,
        collection=args.collection, mode=args.mode, out=args.out,
        reference=args.reference, window_days=args.window, max_frames=args.frames,
    )


def cmd_cap(args) -> int:
    from .api.state import get_engine
    from .decision.graph import TravelMode
    from .delivery import build_cap_alert, validate_cap

    engine = get_engine(downsample=args.downsample)
    _scenario, plan = engine.plan(args.district, mode=TravelMode(args.mode), frame=args.frame)
    alert = build_cap_alert(plan, engine.district(args.district).config)
    problems = validate_cap(alert.xml)
    if args.out:
        Path(args.out).write_text(alert.xml, encoding="utf-8")
        print(f"Written to {args.out}")
    else:
        print(alert.xml)
    print(f"\nValidation: {'VALID' if not problems else problems}", file=sys.stderr)
    return 0 if not problems else 1


def cmd_status(args) -> int:
    from .config import available_districts, load_config
    from .hazard.infer import checkpoint_status
    from .config import models_dir
    from .ingest.weather import feed_status

    print("\n=== PRAHARI STATUS ===\n")
    print("Districts configured:")
    for did in available_districts():
        cfg = load_config(did)
        print(f"  {did:12s} {cfg.name}, {cfg.district.state} ({cfg.district.basin} basin)")

    print("\nHazard model:")
    status = checkpoint_status(models_dir() / "unet_sar_flood.pt")
    if status["usable"]:
        print(f"  U-Net checkpoint present, val IoU {status['val_iou']:.4f}, "
              f"real data: {status['trained_on_real_data']}")
    else:
        print("  No U-Net checkpoint. Serving the classical Otsu baseline.")
        print("  This is the designed fallback, not a failure — train with")
        print("    python -m prahari.hazard.train --data <sen1floods11>")

    print("\nData feeds:")
    for name, info in feed_status().items():
        mark = "OK  " if info["available"] else "----"
        print(f"  [{mark}] {name:32s} {info['note']}")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prahari",
        description="PRAHARI — flood evacuation decision layer. "
                    "Turns a flood warning into a route.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the dashboard server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="synthetic end-to-end run (no network needed)")
    p.add_argument("--district", default="barpeta")
    p.add_argument("--mode", default="foot", choices=["foot", "vehicle"])
    p.add_argument("--stage", type=float, default=3.0, help="river stage in metres")
    p.add_argument("--downsample", type=int, default=3)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("real", help="run on real observations")
    p.add_argument("--district", default="barpeta")
    p.add_argument("--date", default="2024-07-16")
    p.add_argument("--mode", default="foot", choices=["foot", "vehicle"])
    p.add_argument("--collection", default="sentinel-1-rtc")
    p.set_defaults(func=cmd_real)

    p = sub.add_parser("ingest", help="pre-fetch and cache everything (run before demo day)")
    p.add_argument("--district", default="barpeta")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--scenes", type=int, default=3)
    p.add_argument("--collection", default="sentinel-1-rtc")
    p.add_argument("--population", action="store_true",
                   help="also download the WorldPop country raster (~490 MB)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("fetch-population", help="one-time WorldPop country download")
    p.add_argument("--district", default="barpeta")
    p.set_defaults(func=cmd_fetch_population)

    p = sub.add_parser("fetch-training-data", help="download Sen1Floods11 (public, no auth)")
    p.add_argument("--root", default="data/raw/sen1floods11")
    p.set_defaults(func=cmd_fetch_training_data)

    p = sub.add_parser("train", help="train the SAR flood segmentation U-Net")
    p.add_argument("--data", default="data/raw/sen1floods11")
    p.add_argument("--out", default="models")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("validate", help="retrospective validation against a real event")
    p.add_argument("--district", default="barpeta")
    p.add_argument("--date", default="2024-07-16")
    p.add_argument("--collection", default="sentinel-1-rtc")
    p.add_argument("--mode", default="foot", choices=["foot", "vehicle"])
    p.add_argument("--out", default=None)
    p.add_argument("--window", type=int, default=24, help="days either side of the event")
    p.add_argument("--frames", type=int, default=6, help="max acquisitions to process")
    p.add_argument("--reference", default=None,
                   help="independent flood mask GeoTIFF (e.g. a Bhuvan/NDEM export) to score against")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("cap", help="emit a CAP 1.2 alert")
    p.add_argument("--district", default="barpeta")
    p.add_argument("--mode", default="foot", choices=["foot", "vehicle"])
    p.add_argument("--frame", type=int, default=14)
    p.add_argument("--downsample", type=int, default=3)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_cap)

    p = sub.add_parser("status", help="what is configured and what is not")
    p.set_defaults(func=cmd_status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
