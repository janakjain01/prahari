"""FastAPI application — the responder dashboard's backend.

    GET  /                        the dashboard (static, zero build step)
    GET  /api/health              liveness + which hazard model is live
    GET  /api/status              engine introspection
    GET  /api/districts           configured districts
    GET  /api/state               THE endpoint — full dashboard payload
    GET  /api/overlay.png         depth raster as a map overlay
    GET  /api/cap.xml             CAP 1.2 alert
    GET  /api/cap/validate        structural validation of the emitted CAP
    GET  /api/cards               citizen action cards
    GET  /api/replay              replay event metadata (the timeline)
    GET  /api/priorities.csv      boat dispatch list, for a printed handover

Design rules:

  * Every response that describes a flood carries its own DISCLOSURE block. A
    client cannot render this data without also receiving whether it is
    synthetic, replayed or live. That is deliberate — it makes an honest UI
    the path of least resistance and a dishonest one extra work.

  * Errors return real messages. "500 Internal Server Error" on stage is
    worthless; "no shelter is reachable from this settlement" is debuggable in
    front of an audience.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import available_districts, load_config, project_root
from ..decision.graph import TravelMode
from ..delivery.action_card import build_all_cards
from ..delivery.cap import build_cap_alert, validate_cap
from ..delivery.dashboard import build_dashboard_payload, render_depth_png
from ..delivery.i18n import LANGUAGES
from .state import get_engine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PRAHARI — Flood Evacuation Decision Layer",
    description=(
        "India can already predict floods. It still cannot tell a family which road "
        "is safe. PRAHARI is the decision layer that turns a flood warning into a route."
    ),
    version=__version__,
)

# The dashboard is served from the same origin, but CORS is open so the API can
# be driven from a separate frontend during development or from a judge's own
# tooling.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

WEB_DIR = project_root() / "web"


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "version": __version__,
        "service": "prahari",
        "hazard_model": engine.status()["hazard_model"],
    }


@app.get("/api/status")
def status() -> dict:
    return get_engine().status()


@app.get("/api/districts")
def districts() -> dict:
    out = []
    for did in available_districts():
        cfg = load_config(did)
        out.append(
            {
                "id": cfg.id,
                "name": cfg.name,
                "state": cfg.district.state,
                "basin": cfg.district.basin,
                "bbox": cfg.grid.bbox.as_tuple(),
                "centroid": cfg.grid.bbox.centroid,
                "languages": cfg.alerting.languages,
            }
        )
    return {"districts": out}


# --------------------------------------------------------------------------
# The main endpoint
# --------------------------------------------------------------------------
@app.get("/api/state")
def state(
    district: str = Query("barpeta"),
    mode: Literal["foot", "vehicle"] = Query("foot"),
    source: Literal["replay", "synthetic", "real"] = Query("replay"),
    frame: int = Query(0, ge=0),
    stage_m: Optional[float] = Query(None, ge=0, le=15),
    imminent_band_m: float = Query(0.0, ge=0, le=3),
    include_clear_roads: bool = Query(True),
    include_layers: bool = Query(True),
    date: Optional[str] = Query(None, description="acquisition date for source=real, YYYY-MM-DD"),
    detector: Literal["otsu", "unet"] = Query(
        "otsu",
        description=(
            "Hazard detector for source=real. Defaults to the Otsu baseline: it is "
            "physically grounded, terrain-vetoed and errs toward MORE water. The U-Net "
            "wins on Sen1Floods11 but disagrees by 2.5x on real Indian scenes and is "
            "not yet validated against ground truth for Assam."
        ),
    ),
) -> JSONResponse:
    """Full dashboard payload for one flood state."""
    engine = get_engine()
    try:
        scenario, plan = engine.plan(
            district_id=district,
            mode=TravelMode(mode),
            source=source,
            frame=frame,
            stage_m=stage_m,
            imminent_band_m=imminent_band_m,
            date=date,
            detector=detector,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        # Real-data build failures carry a useful message; surface it rather
        # than a bare 500 that is undebuggable in front of an audience.
        raise HTTPException(503, str(exc)) from exc

    payload = build_dashboard_payload(
        scenario, plan, load_config(district),
        include_layers=include_layers,
        include_clear_roads=include_clear_roads,
    )
    payload["request"] = {
        "district": district, "mode": mode, "source": source,
        "frame": frame, "stage_m": stage_m, "imminent_band_m": imminent_band_m,
    }
    return JSONResponse(payload)


@app.get("/api/overlay.png")
def overlay(
    district: str = Query("barpeta"),
    source: Literal["replay", "synthetic", "real"] = Query("replay"),
    frame: int = Query(0, ge=0),
    stage_m: Optional[float] = Query(None, ge=0, le=15),
    max_side: int = Query(1600, ge=256, le=4096),
    date: Optional[str] = Query(None),
    detector: Literal["otsu", "unet"] = Query("otsu"),
) -> Response:
    """Flood depth as a transparent PNG, for a MapLibre image source."""
    engine = get_engine()
    scenario, _plan = engine.plan(
        district_id=district, source=source, frame=frame, stage_m=stage_m,
        date=date, detector=detector,
    )
    try:
        png = render_depth_png(scenario, max_side=max_side)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=60"},
    )


# --------------------------------------------------------------------------
# Delivery
# --------------------------------------------------------------------------
@app.get("/api/cap.xml")
def cap_xml(
    district: str = Query("barpeta"),
    mode: Literal["foot", "vehicle"] = Query("foot"),
    source: Literal["replay", "synthetic", "real"] = Query("replay"),
    frame: int = Query(0, ge=0),
    stage_m: Optional[float] = Query(None, ge=0, le=15),
) -> Response:
    """CAP 1.2 alert — the same schema SACHET consumes."""
    engine = get_engine()
    _scenario, plan = engine.plan(
        district_id=district, mode=TravelMode(mode), source=source, frame=frame, stage_m=stage_m
    )
    alert = build_cap_alert(plan, load_config(district))
    return Response(
        content=alert.xml,
        media_type="application/xml",
        headers={"Content-Disposition": f'inline; filename="{alert.identifier}.xml"'},
    )


@app.get("/api/cap/validate")
def cap_validate(
    district: str = Query("barpeta"),
    mode: Literal["foot", "vehicle"] = Query("foot"),
    source: Literal["replay", "synthetic", "real"] = Query("replay"),
    frame: int = Query(0, ge=0),
) -> dict:
    """Validate our own CAP output. Useful to run in front of a judge."""
    engine = get_engine()
    _scenario, plan = engine.plan(
        district_id=district, mode=TravelMode(mode), source=source, frame=frame
    )
    alert = build_cap_alert(plan, load_config(district))
    problems = validate_cap(alert.xml)
    return {
        "identifier": alert.identifier,
        "valid": not problems,
        "problems": problems,
        "checks": [
            "mandatory <alert> elements present",
            "mandatory <info> elements present",
            "every <area> has <areaDesc>",
            "polygon rings are closed",
            "coordinates are lat,lon and in range (CAP order, not GeoJSON order)",
            "status is not 'Actual' in a prototype",
        ],
        **alert.to_dict(),
    }


@app.get("/api/cards")
def cards(
    district: str = Query("barpeta"),
    mode: Literal["foot", "vehicle"] = Query("foot"),
    source: Literal["replay", "synthetic", "real"] = Query("replay"),
    frame: int = Query(0, ge=0),
    language: Optional[str] = Query(None),
    settlement: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=500),
) -> dict:
    """Citizen action cards."""
    engine = get_engine()
    _scenario, plan = engine.plan(
        district_id=district, mode=TravelMode(mode), source=source, frame=frame
    )
    languages = [language] if language else list(LANGUAGES)
    unknown = [lang for lang in languages if lang not in LANGUAGES]
    if unknown:
        raise HTTPException(400, f"unsupported language(s) {unknown}; available: {list(LANGUAGES)}")

    built = build_all_cards(plan, languages=languages, limit=limit)
    if settlement:
        built = [c for c in built if c.settlement_id == settlement]
        if not built:
            raise HTTPException(404, f"no card for settlement '{settlement}' in this state")

    return {
        "count": len(built),
        "languages": languages,
        "translation_caveat": (
            "Assamese, Bengali and Hindi strings were prepared by the team and have NOT "
            "been validated by a native speaker or by ASDMA. Alert wording is a "
            "life-safety surface and must be reviewed before any real deployment."
        ),
        "cards": [{**c.to_dict(), "text": c.to_text()} for c in built],
    }


@app.get("/api/priorities.csv")
def priorities_csv(
    district: str = Query("barpeta"),
    mode: Literal["foot", "vehicle"] = Query("foot"),
    source: Literal["replay", "synthetic", "real"] = Query("replay"),
    frame: int = Query(0, ge=0),
    only_cut_off: bool = Query(False),
) -> Response:
    """Dispatch list as CSV.

    Exists because a district control room runs on paper and WhatsApp during
    an event, not on a web dashboard with a live connection. A printable
    handover is a real requirement, not a nice-to-have.
    """
    engine = get_engine()
    _scenario, plan = engine.plan(
        district_id=district, mode=TravelMode(mode), source=source, frame=frame
    )
    entries = plan.boat_dispatch_list() if only_cut_off else plan.priorities

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["rank", "settlement", "reason", "people_at_risk", "population",
         "depth_m", "lat", "lon", "nearest_shelter_minutes", "action"]
    )
    for e in entries:
        writer.writerow(
            [e.rank, e.settlement_name, e.primary_reason.value, e.people_at_risk,
             e.population, round(e.depth_m, 2), round(e.lat, 6), round(e.lon, 6),
             "" if e.nearest_shelter_minutes is None else round(e.nearest_shelter_minutes, 1),
             e.action]
        )
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="prahari_dispatch_{district}.csv"'},
    )


@app.get("/api/replay")
def replay(district: str = Query("barpeta")) -> dict:
    """Replay timeline metadata."""
    engine = get_engine()
    state_obj = engine.district(district)
    if state_obj.event is None:
        raise HTTPException(404, f"no replay event configured for {district}")
    return state_obj.event.to_dict()


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
else:  # pragma: no cover

    @app.get("/", response_class=PlainTextResponse)
    def index_missing() -> str:
        return (
            "PRAHARI API is running, but the web/ directory was not found.\n"
            "The API is fully usable — try /docs or /api/state?district=barpeta\n"
        )


def main() -> None:  # pragma: no cover
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the PRAHARI dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--downsample", type=int, default=2,
        help="Grid coarsening factor. 1 = full 30 m resolution (slow first build), "
             "2 = 60 m (recommended for a live demo), 3 = 90 m (fastest).",
    )
    args = parser.parse_args()

    import os

    os.environ["PRAHARI_DOWNSAMPLE"] = str(args.downsample)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "prahari.api.main:app" if args.reload else app,
        host=args.host, port=args.port, reload=args.reload,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
