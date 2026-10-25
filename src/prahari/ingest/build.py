"""Assemble a REAL district scenario from live sources.

    build_real_scenario(config) -> (Scenario, report)

Every fetch is allowed to fail. The report records exactly what came from
where and what did not, and that report is surfaced in the UI rather than
logged and forgotten.

Degradation ladder, in order:

    DEM          required. Without elevation there is no HAND, and without
                 HAND there is no depth. This is the one hard dependency, and
                 it is on a free, keyless, waitlist-free source.
    roads        required. Without a network there is nothing to route on.
    population   falls back to a uniform density over built-up area, clearly
                 flagged, so the pipeline still demonstrates end to end.
    shelters     falls back to nothing, which is honest: the plan then shows
                 the entire population as unshelterable, which is exactly
                 what a district with no mapped shelters means.
    SAR          optional at build time; the hazard layer is applied later.

"Required" here means the function returns a failure report rather than a
half-built scenario. It never returns something that looks complete and is not.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from ..config import DistrictConfig
from ..grid import Grid, grid_from_config
from ..scenario import Scenario, Settlement
from .base import IngestResult
from .dem import burn_embankments, fetch_dem
from .osm import (
    fetch_embankments,
    fetch_place_names,
    fetch_road_network,
    fetch_shelter_candidates,
    label_settlements,
)
from .population import fetch_worldpop

logger = logging.getLogger(__name__)

__all__ = ["build_real_scenario", "BuildReport"]


class BuildReport(dict):
    """Per-layer ingest outcomes, plus whether the scenario is usable."""

    @property
    def ok(self) -> bool:
        return bool(self.get("usable"))

    def summary(self) -> str:
        lines = [f"Scenario build: {'USABLE' if self.ok else 'FAILED'}"]
        for layer, info in self.get("layers", {}).items():
            mark = "ok " if info.get("ok") else "FAIL"
            detail = info.get("detail") or info.get("reason") or ""
            lines.append(f"  [{mark}] {layer:12s} {detail}")
        return "\n".join(lines)


def build_real_scenario(
    config: DistrictConfig,
    use_cache: bool = True,
    burn_dykes: bool = True,
    max_settlements: int = 400,
    allow_population_download: bool = False,
) -> tuple[Optional[Scenario], BuildReport]:
    """Build a district scenario from real observations."""
    from ..hazard.hand import compute_hand
    from ..impact.population import derive_settlements

    grid = grid_from_config(config)
    bbox = config.grid.bbox.as_tuple()
    district_id = config.district.id
    report = BuildReport(layers={}, district=district_id, usable=False)

    def record(layer: str, result: IngestResult) -> None:
        report["layers"][layer] = result.to_dict()

    logger.info("Building REAL scenario for %s on %s", config.name, grid)

    # --- 1. Elevation (hard requirement) ---------------------------------
    dem_result = fetch_dem(
        grid, bbox,
        collection=config.sources.dem.get("stac_collection", "cop-dem-glo-30"),
        district_id=district_id, use_cache=use_cache,
    )
    record("dem", dem_result)
    if not dem_result:
        report["fatal"] = (
            "No elevation data. HAND, and therefore every depth estimate, is impossible. "
            "Check network access to planetarycomputer.microsoft.com."
        )
        return None, report

    dem = np.asarray(dem_result.data, dtype=np.float64)

    # --- 2. Embankments (optional, but material in Assam) ----------------
    if burn_dykes:
        dyke_result = fetch_embankments(bbox, grid)
        record("embankments", dyke_result)
        if dyke_result:
            dem = burn_embankments(dem, grid, dyke_result.data)

    # --- 3. HAND ---------------------------------------------------------
    hand_result = compute_hand(
        dem,
        cell_size=grid.resolution,
        drainage_threshold_cells=config.hydrology.threshold_cells(grid.resolution),
        fill=config.hydrology.fill_depressions,
    )
    report["layers"]["hand"] = {
        "ok": True,
        "detail": (
            f"HAND from DEM, channel threshold "
            f"{config.hydrology.drainage_threshold_km2:g} km2, "
            f"{100 * hand_result.drainage.mean():.2f}% of cells are channel"
        ),
    }

    # --- 4. Roads (hard requirement) -------------------------------------
    roads_result = fetch_road_network(
        bbox, grid,
        network_type=config.sources.osm.get("road_network_type", "drive"),
        district_id=district_id, use_cache=use_cache,
    )
    record("roads", roads_result)
    if not roads_result:
        report["fatal"] = "No road network. There is nothing to route on."
        return None, report

    # --- 5. Population ---------------------------------------------------
    # allow_full_download stays OFF here on purpose: a routine scenario build
    # must never silently start a half-gigabyte download. Pre-fetch it
    # deliberately with `prahari fetch-population` during preparation.
    pop_result = fetch_worldpop(
        grid, bbox,
        year=int(config.sources.population.get("year", 2020)),
        district_id=district_id, use_cache=use_cache,
        allow_full_download=allow_population_download,
    )
    record("population", pop_result)
    if pop_result:
        population = np.asarray(pop_result.data, dtype=np.float32)
        pop_synthetic = False
    else:
        logger.warning("WorldPop unavailable — substituting a flat placeholder density")
        population = np.full(grid.shape, 1.0, dtype=np.float32)
        pop_synthetic = True

    # --- 6. Shelters -----------------------------------------------------
    shelter_result = fetch_shelter_candidates(bbox, grid, config, district_id=district_id)
    record("shelters", shelter_result)
    shelters = shelter_result.data if shelter_result else []

    # --- 7. Settlements from the population surface ----------------------
    settlements: list[Settlement] = derive_settlements(
        grid, population, max_settlements=max_settlements
    )

    # Give the clusters real village names. "Settlement 108" is useless on a
    # dispatch list — a boat crew needs a name they recognise.
    naming = {"exact": 0, "relative": 0, "generic": len(settlements)}
    places_result = fetch_place_names(bbox, grid)
    record("place_names", places_result)
    if places_result and settlements:
        naming = label_settlements(settlements, places_result.data)

    report["layers"]["settlements"] = {
        "ok": bool(settlements),
        "detail": (
            f"{len(settlements)} clustered from population; names: "
            f"{naming['exact']} exact OSM, {naming['relative']} relative bearing, "
            f"{naming['generic']} generic"
        ),
    }

    scenario = Scenario(
        district_id=district_id,
        grid=grid,
        dem=dem.astype(np.float32),
        hand=hand_result.hand,
        population=population,
        roads=roads_result.data,
        shelters=list(shelters),
        settlements=settlements,
        timestamp=datetime.now(timezone.utc).isoformat(),
        meta={"synthetic": False, "build_report": dict(report)},
    )

    scenario.add_provenance(
        "dem", dem_result.source, detail=dem_result.detail, is_estimate=False
    )
    scenario.add_provenance(
        "hand", "computed",
        detail=f"HAND (Nobre et al. 2011) from {dem_result.source}", is_estimate=True,
    )
    scenario.add_provenance(
        "roads", roads_result.source, detail=roads_result.detail,
    )
    scenario.add_provenance(
        "population",
        pop_result.source if pop_result else "placeholder",
        detail=(
            pop_result.detail if pop_result
            else "WorldPop unavailable — FLAT PLACEHOLDER DENSITY, headcounts are meaningless"
        ),
        is_estimate=True, is_synthetic=pop_synthetic,
    )
    scenario.add_provenance(
        "shelters",
        shelter_result.source if shelter_result else "none",
        detail=(
            shelter_result.detail if shelter_result
            else "No shelters mapped in OSM for this district — every person will report "
                 "as unshelterable until a relief-camp register is supplied"
        ),
        is_estimate=True,
    )

    report["usable"] = True
    report["summary"] = scenario.summary()
    logger.info("REAL scenario for %s ready:\n%s", config.name, report.summary())
    return scenario, report
