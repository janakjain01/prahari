"""Exposure — how many people are standing in water, and where they are.

Two products:

    compute_exposure()    exposed population, aggregated to reporting cells
    derive_settlements()  population raster -> discrete places with demand

The second exists because the decision layer needs SOURCES: a min-cost flow
formulation needs supply nodes with a population, not a continuous density
raster. Where OSM place nodes are available we use them; where they are not —
which is most of rural Assam at village granularity — we cluster the
population raster into settlements.

Honesty note carried through to the UI: WorldPop is a modelled product, not a
census enumeration. It disaggregates census totals using built-up area and
other covariates. It is very good for "roughly how many people are in this
square kilometre" and should not be quoted to the individual.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..grid import Grid
from ..scenario import Settlement

logger = logging.getLogger(__name__)

__all__ = [
    "compute_exposure",
    "ExposureResult",
    "derive_settlements",
    "ExposureCell",
    "compute_evacuation_demand",
]


@dataclass
class ExposureCell:
    """One reporting cell — the granularity a district officer can act on."""

    id: str
    x: float
    y: float
    lon: float
    lat: float
    population: float
    exposed: float
    max_depth_m: float
    mean_depth_m: float

    @property
    def exposed_fraction(self) -> float:
        return self.exposed / self.population if self.population > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "lon": round(self.lon, 6),
            "lat": round(self.lat, 6),
            "population": round(self.population, 1),
            "exposed": round(self.exposed, 1),
            "exposed_fraction": round(self.exposed_fraction, 4),
            "max_depth_m": round(self.max_depth_m, 2),
            "mean_depth_m": round(self.mean_depth_m, 2),
        }


@dataclass
class ExposureResult:
    total_population: float
    exposed_population: float
    flooded_area_km2: float
    cells: list[ExposureCell]
    cell_size_m: float

    @property
    def exposed_fraction(self) -> float:
        return self.exposed_population / self.total_population if self.total_population else 0.0

    def to_dict(self) -> dict:
        return {
            "total_population": round(self.total_population),
            "exposed_population": round(self.exposed_population),
            "exposed_fraction": round(self.exposed_fraction, 4),
            "flooded_area_km2": round(self.flooded_area_km2, 2),
            "cell_size_m": self.cell_size_m,
            "n_cells": len(self.cells),
        }


def compute_exposure(
    grid: Grid,
    population: np.ndarray,
    flood_mask: np.ndarray,
    depth: np.ndarray,
    cell_size_m: float = 1000.0,
    min_exposed: float = 1.0,
) -> ExposureResult:
    """Aggregate exposed population into square reporting cells.

    Reporting at 1 km rather than at native 30 m resolution is a deliberate
    choice, and it is about honesty as much as legibility: WorldPop's own
    input resolution is ~100 m and it is a modelled surface, so presenting a
    per-30-m-pixel headcount would imply a precision the data does not have.
    1 km is roughly the granularity at which a district officer dispatches
    anyway.
    """
    pop = np.nan_to_num(np.asarray(population, dtype=np.float64))
    mask = np.asarray(flood_mask) > 0
    dep = np.nan_to_num(np.asarray(depth, dtype=np.float64))

    exposed_raster = pop * mask
    block = max(int(round(cell_size_m / grid.resolution)), 1)

    h, w = grid.shape
    cells: list[ExposureCell] = []

    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            r1, c1 = min(r0 + block, h), min(c0 + block, w)
            block_pop = float(pop[r0:r1, c0:c1].sum())
            block_exposed = float(exposed_raster[r0:r1, c0:c1].sum())
            if block_exposed < min_exposed:
                continue
            block_depth = dep[r0:r1, c0:c1][mask[r0:r1, c0:c1]]
            cx, cy = grid.xy((r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0)
            lon, lat = grid.to_geographic(cx, cy)
            cells.append(
                ExposureCell(
                    id=f"cell_{r0 // block:03d}_{c0 // block:03d}",
                    x=float(cx), y=float(cy),
                    lon=float(lon), lat=float(lat),
                    population=block_pop,
                    exposed=block_exposed,
                    max_depth_m=float(block_depth.max()) if block_depth.size else 0.0,
                    mean_depth_m=float(block_depth.mean()) if block_depth.size else 0.0,
                )
            )

    cells.sort(key=lambda c: -c.exposed)
    result = ExposureResult(
        total_population=float(pop.sum()),
        exposed_population=float(exposed_raster.sum()),
        flooded_area_km2=float(mask.sum()) * grid.cell_area_m2 / 1e6,
        cells=cells,
        cell_size_m=cell_size_m,
    )
    logger.info(
        "Exposure: %.0f of %.0f people in water (%.1f%%) across %.1f km2, %d reporting cells",
        result.exposed_population, result.total_population,
        100 * result.exposed_fraction, result.flooded_area_km2, len(cells),
    )
    return result


def compute_evacuation_demand(
    grid: Grid,
    population: np.ndarray,
    flood_mask: np.ndarray,
    settlements: Sequence[Settlement],
    depth: Optional[np.ndarray] = None,
    at_risk_depth_m: float = 0.0,
    include_imminent_band_m: float = 0.0,
    hand: Optional[np.ndarray] = None,
) -> float:
    """Work out how many people each settlement actually has to move.

    THIS IS A CORRECTNESS ISSUE, NOT A REFINEMENT. Handing the assignment
    every person in the district asks the optimiser to evacuate 1.7 million
    people into 50,000 shelter places, reports 97% of them as "unsheltered",
    and buries the real answer. You do not evacuate a district; you evacuate
    the part of it that is under water or about to be.

    Method: every flooded population pixel is attributed to its NEAREST
    settlement — a Voronoi partition of the floodplain — and the attributed
    people become that settlement's demand. Voronoi rather than a fixed radius
    because village catchments are not circles of equal size, and a fixed
    radius either double-counts overlapping villages or drops the people
    between them.

    ``include_imminent_band_m``
        Optionally also counts people who are DRY BUT LOW — within this many
        metres of the current water surface on the HAND datum. Evacuating only
        those already standing in water is evacuating too late. Off by default
        so the headline number stays strictly observational; turn it on for
        forward-looking planning and say which mode is running.

    Returns the total demand, and sets ``settlement.people_at_risk`` in place.
    """
    from scipy import ndimage

    pop = np.nan_to_num(np.asarray(population, dtype=np.float64))
    at_risk_mask = np.asarray(flood_mask) > 0

    if depth is not None and at_risk_depth_m > 0:
        at_risk_mask = at_risk_mask & (np.nan_to_num(depth) >= at_risk_depth_m)

    if include_imminent_band_m > 0 and hand is not None and np.any(flood_mask):
        # Water surface on the HAND datum ~= the highest HAND currently wet.
        wet_hand = np.asarray(hand)[np.asarray(flood_mask) > 0]
        wet_hand = wet_hand[np.isfinite(wet_hand)]
        if wet_hand.size:
            surface = float(np.percentile(wet_hand, 90))
            imminent = (np.asarray(hand) > surface) & (
                np.asarray(hand) <= surface + include_imminent_band_m
            )
            at_risk_mask = at_risk_mask | imminent

    for s in settlements:
        s.people_at_risk = 0.0

    if not at_risk_mask.any() or not settlements:
        return 0.0

    # Seed raster: one pixel per settlement, then nearest-seed attribution.
    seeds = np.zeros(grid.shape, dtype=bool)
    seed_index: dict[tuple[int, int], int] = {}
    for i, s in enumerate(settlements):
        row, col = grid.rowcol(s.x, s.y, clip=True)
        r, c = int(row), int(col)
        seeds[r, c] = True
        seed_index[(r, c)] = i

    _dist, (ri, ci) = ndimage.distance_transform_edt(~seeds, return_indices=True)

    flooded_pop = pop * at_risk_mask
    rows, cols = np.nonzero(flooded_pop > 0)
    if rows.size == 0:
        return 0.0

    owner_r = ri[rows, cols]
    owner_c = ci[rows, cols]
    values = flooded_pop[rows, cols]

    totals = np.zeros(len(settlements), dtype=np.float64)
    # Vectorised bincount over the flattened owner index.
    owner_flat = owner_r.astype(np.int64) * grid.width + owner_c.astype(np.int64)
    unique_owners, inverse = np.unique(owner_flat, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=unique_owners.size)
    for flat, total in zip(unique_owners, sums):
        r, c = int(flat // grid.width), int(flat % grid.width)
        idx = seed_index.get((r, c))
        if idx is not None:
            totals[idx] += total

    for s, total in zip(settlements, totals):
        # Cannot need to move more people than live there.
        s.people_at_risk = float(min(total, s.population))

    demand = float(sum(s.people_at_risk for s in settlements))
    logger.info(
        "Evacuation demand: %.0f people at risk across %d settlements (%.1f%% of the "
        "district population of %.0f)",
        demand,
        sum(1 for s in settlements if s.people_at_risk > 0),
        100 * demand / max(sum(s.population for s in settlements), 1),
        sum(s.population for s in settlements),
    )
    return demand


def derive_settlements(
    grid: Grid,
    population: np.ndarray,
    min_population: float = 50.0,
    density_percentile: float = 88.0,
    smooth_sigma_px: float = 2.0,
    max_settlements: int = 400,
    names: Optional[dict[str, str]] = None,
) -> list[Settlement]:
    """Cluster a population raster into discrete settlements.

    Method: smooth, threshold at a high density percentile, label connected
    components, and take each component's population-weighted centroid. The
    smoothing matters — without it, WorldPop's pixel-level noise fragments one
    village into a dozen "settlements", which then compete against each other
    for shelter capacity in the assignment and produce nonsense.

    ``max_settlements`` caps the problem size handed to the min-cost flow.
    Components are ranked by population, so the cap drops the smallest hamlets
    first; their people are NOT discarded — see the reassignment of residual
    population below.
    """
    from scipy import ndimage

    pop = np.nan_to_num(np.asarray(population, dtype=np.float64))
    if pop.sum() <= 0:
        logger.warning("Population raster is empty — no settlements derived")
        return []

    smoothed = ndimage.gaussian_filter(pop, sigma=smooth_sigma_px)
    positive = smoothed[smoothed > 0]
    if positive.size == 0:
        return []
    threshold = float(np.percentile(positive, density_percentile))
    if threshold <= 0:
        threshold = float(positive.mean())

    labels, n = ndimage.label(smoothed >= threshold)
    if n == 0:
        logger.warning("No settlement clusters found above the density threshold")
        return []

    settlements: list[Settlement] = []
    for idx in range(1, n + 1):
        comp = labels == idx
        comp_pop = float(pop[comp].sum())
        if comp_pop < min_population:
            continue
        rows, cols = np.nonzero(comp)
        weights = pop[rows, cols]
        if weights.sum() <= 0:
            continue
        r = float((rows * weights).sum() / weights.sum())
        c = float((cols * weights).sum() / weights.sum())
        x, y = grid.xy(r, c)
        settlements.append(
            Settlement(
                id=f"s{idx:04d}",
                name=f"Settlement {idx:03d}",
                x=float(x), y=float(y),
                population=comp_pop,
            )
        )

    settlements.sort(key=lambda s: -s.population)

    if len(settlements) > max_settlements:
        # Do not silently drop people. Redistribute the tail's population
        # across the nearest retained settlements so the total demand handed to
        # the assignment still matches the district population.
        dropped = settlements[max_settlements:]
        settlements = settlements[:max_settlements]
        kept_xy = np.array([[s.x, s.y] for s in settlements])
        for d in dropped:
            dist2 = ((kept_xy[:, 0] - d.x) ** 2 + (kept_xy[:, 1] - d.y) ** 2)
            settlements[int(np.argmin(dist2))].population += d.population
        logger.info(
            "Capped at %d settlements; %d smaller clusters folded into their nearest neighbour "
            "(no population lost)", max_settlements, len(dropped),
        )

    if names:
        for s in settlements:
            if s.id in names:
                s.name = names[s.id]

    # A settlement point represents its CATCHMENT, not just its dense core.
    #
    # Clustering picks out high-density cores, which on a real WorldPop raster
    # is only a fraction of the district — 377k of 1.4M people in Barpeta. The
    # rest live dispersed between the cores. If each settlement carries only its
    # core population, and evacuation demand is capped at that figure, then most
    # flooded people are attributed to no settlement at all and simply vanish
    # from the plan: 223,091 people observed inside the flood footprint but only
    # 49,124 counted as needing evacuation.
    #
    # Assigning every populated pixel to its nearest settlement fixes that — the
    # settlement totals then reconcile with the district population, and the
    # demand model has somewhere to put everyone.
    _assign_catchment_population(grid, pop, settlements)

    covered = sum(s.population for s in settlements)
    logger.info(
        "Settlements: %d places covering %.0f of %.0f people (%.0f%%)",
        len(settlements), covered, pop.sum(), 100 * covered / max(pop.sum(), 1),
    )
    return settlements


def _assign_catchment_population(
    grid: Grid, population: np.ndarray, settlements: Sequence[Settlement]
) -> None:
    """Give each settlement the total population of its Voronoi catchment."""
    from scipy import ndimage

    if not settlements:
        return

    seeds = np.zeros(grid.shape, dtype=bool)
    seed_index: dict[tuple[int, int], int] = {}
    for i, s in enumerate(settlements):
        row, col = grid.rowcol(s.x, s.y, clip=True)
        r, c = int(row), int(col)
        seeds[r, c] = True
        seed_index[(r, c)] = i

    _dist, (ri, ci) = ndimage.distance_transform_edt(~seeds, return_indices=True)

    rows, cols = np.nonzero(population > 0)
    if rows.size == 0:
        return
    owner_flat = ri[rows, cols].astype(np.int64) * grid.width + ci[rows, cols].astype(np.int64)
    values = population[rows, cols]

    unique_owners, inverse = np.unique(owner_flat, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=unique_owners.size)

    totals = np.zeros(len(settlements), dtype=np.float64)
    for flat, total in zip(unique_owners, sums):
        idx = seed_index.get((int(flat // grid.width), int(flat % grid.width)))
        if idx is not None:
            totals[idx] += total

    for s, total in zip(settlements, totals):
        s.population = float(total)
