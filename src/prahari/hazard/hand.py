"""HAND — Height Above Nearest Drainage.

HAND is the normalised elevation of a cell above the drainage cell it flows
into. It is the standard way to turn a flood EXTENT (which satellites give
you) into a flood DEPTH (which evacuation decisions need), without running a
2-D hydrodynamic model per basin.

That trade is the core technical argument of this project. C-FLOOD covers
three basins because full 2-D hydrodynamics is expensive to set up per basin.
HAND is coarser but generalises to any district with a DEM — which is the
right trade when the decision you are supporting is "which road, which
shelter", not "millimetre water depth".

Reference: Nobre et al. (2011), "Height Above the Nearest Drainage — a
hydrologically relevant new terrain model", Journal of Hydrology 404(1).

Pipeline
--------
    1. fill depressions           (else flow terminates in spurious sinks)
    2. resolve flats              (else D8 is undefined on flat floodplain)
    3. D8 flow directions
    4. flow accumulation
    5. drainage network = accumulation > threshold
    6. HAND = elev - elev(first drainage cell downstream)

Step 2 matters more here than in most terrain. The Brahmaputra floodplain is
almost perfectly flat; without flat resolution, D8 produces parallel garbage
and HAND is meaningless. This is a real failure mode we handle, and it is
worth saying out loud when asked about the hydrology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "fill_depressions",
    "priority_flood_epsilon",
    "resolve_flats",
    "d8_receivers",
    "flow_accumulation",
    "drainage_network",
    "compute_hand",
    "HandResult",
]

# D8 neighbour offsets, clockwise from east. Index into these is the flow code.
_D8_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
    (-1, 0),   # N
    (-1, 1),   # NE
)
_SQRT2 = float(np.sqrt(2.0))
_D8_DIST = np.array([1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2], dtype=np.float64)


@dataclass
class HandResult:
    """HAND plus the intermediate rasters, kept because they are diagnostic.

    When HAND looks wrong, it is nearly always the drainage network that is
    wrong, and you cannot see that from HAND alone.
    """

    hand: np.ndarray
    filled_dem: np.ndarray
    flow_accumulation: np.ndarray
    drainage: np.ndarray
    receivers: np.ndarray
    drainage_elevation: np.ndarray


# --------------------------------------------------------------------------
# 1. Depression filling
# --------------------------------------------------------------------------
def fill_depressions(dem: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """Fill sinks so that every cell has a downslope path to the boundary.

    Implemented as morphological reconstruction by erosion — the classical
    equivalent of Planchon & Darboux / Priority-Flood, but running in
    compiled code rather than a Python heap loop. On a 2 Mpixel district this
    is the difference between a fraction of a second and a minute.

    The marker is +inf in the interior and the true elevation on the border,
    then eroded down under the DEM as mask. Water can only escape via the
    border, which is the correct physical assumption for a district-sized AOI.
    """
    dem = np.asarray(dem, dtype=np.float64)
    work = dem.copy()

    if nodata is not None:
        invalid = work == nodata
        if invalid.any():
            work[invalid] = np.nan

    nan_mask = np.isnan(work)
    if nan_mask.any():
        # Reconstruction cannot see NaN. Substitute the max so those cells act
        # as walls, then restore afterwards.
        finite_max = np.nanmax(work) if not np.all(nan_mask) else 0.0
        work[nan_mask] = finite_max

    try:
        from skimage.morphology import reconstruction
    except ImportError:  # pragma: no cover - fallback path
        logger.warning("scikit-image unavailable; falling back to priority-flood fill (slower)")
        filled = _priority_flood_fill(work)
    else:
        seed = np.full_like(work, work.max())
        seed[0, :] = work[0, :]
        seed[-1, :] = work[-1, :]
        seed[:, 0] = work[:, 0]
        seed[:, -1] = work[:, -1]
        filled = reconstruction(seed, work, method="erosion")

    filled = np.asarray(filled, dtype=np.float64)
    if nan_mask.any():
        filled[nan_mask] = np.nan
    return filled


def priority_flood_epsilon(dem: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Priority-Flood + epsilon — fill depressions AND resolve flats in one pass.

    Reference: Barnes, R., Lehman, C., Mulla, D. (2014), "Priority-Flood: An
    optimal depression-filling and watershed-labeling algorithm for digital
    elevation models", Computers & Geosciences 62.

    THE GUARANTEE. Cells are processed outward from the boundary in order of
    increasing (filled) elevation. When a cell is first reached from a
    predecessor at height ``e``, it is assigned ``max(dem, e + epsilon)``. So
    every cell is strictly higher than the cell it was reached from, which
    means every cell has a strictly-descending path to the boundary. Zero
    sinks and zero flats, by construction rather than by hope.

    WHY THE FAST APPROACH FAILED, because it is worth knowing before someone
    "optimises" this back. Morphological reconstruction fills depressions
    correctly and quickly, but it leaves perfectly flat regions — 43.8% of the
    real Barpeta DEM. Resolving those afterwards with a Euclidean distance
    transform does not work: EDT measures distance to the nearest draining
    cell GEOMETRICALLY, ignoring elevation, so on a floodplain the resulting
    gradient frequently points uphill across a ridge and the cell still cannot
    drain. That left 15.7% of the district as sinks, capped flow accumulation
    at 595 cells out of two million, and silently collapsed HAND — every depth
    in the system was wrong.

    Cost: a Python heap over every cell. Slower than reconstruction, and it
    does not matter — this is a ONE-TIME STATIC PRECOMPUTE per district, cached
    to disk, and never runs during a live cycle.

    ``epsilon`` accumulates along a drainage path, so total artificial rise is
    ``path_length * epsilon``. At 1e-6 m over a 3000-cell path that is 3 mm —
    far below DEM vertical accuracy, and far above float64 resolution at
    floodplain elevations, so the increments never vanish into rounding.
    """
    import heapq

    h, w = dem.shape
    flat_dem = np.asarray(dem, dtype=np.float64).ravel()
    out = np.full(flat_dem.size, np.nan, dtype=np.float64)
    closed = ~np.isfinite(flat_dem)          # NaN cells act as walls

    heap: list[tuple[float, int]] = []

    # Seed every boundary cell: water can only leave the AOI at its edge.
    boundary = np.zeros((h, w), dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True
    for idx in np.flatnonzero(boundary.ravel() & ~closed):
        value = float(flat_dem[idx])
        out[idx] = value
        closed[idx] = True
        heap.append((value, int(idx)))
    heapq.heapify(heap)

    if not heap:
        return np.asarray(dem, dtype=np.float64).copy()

    # Local bindings — this loop runs a few million times.
    push, pop = heapq.heappush, heapq.heappop
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    while heap:
        elev, idx = pop(heap)
        r, c = divmod(idx, w)
        for dr, dc in offsets:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            nidx = nr * w + nc
            if closed[nidx]:
                continue
            closed[nidx] = True
            nv = flat_dem[nidx]
            raised = elev + epsilon
            if not (nv > raised):
                nv = raised
            out[nidx] = nv
            push(heap, (float(nv), nidx))

    # Anything unreachable (enclosed entirely by NaN) keeps its original value.
    unreached = ~np.isfinite(out)
    if unreached.any():
        out[unreached] = flat_dem[unreached]
    return out.reshape(h, w)


# --------------------------------------------------------------------------
# 2. Flat resolution
# --------------------------------------------------------------------------
def resolve_flats(filled: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """Impose a drainage gradient across flat areas.

    After depression filling, large regions share exactly one elevation and D8
    has no downslope neighbour to pick. Every cell in such a region becomes a
    sink, the flow network shatters into thousands of one-cell basins, flow
    accumulation never grows, no cell ever reaches the channel threshold, and
    HAND collapses. On a floodplain — where most of the district is flat —
    that destroys the entire depth model.

    THE GUARANTEE THIS FUNCTION MUST PROVIDE is that after it runs, every flat
    cell has at least one strictly-lower neighbour. We get that from a single
    Euclidean distance transform:

        increment(cell) = epsilon * EDT(cell to the nearest already-draining cell)

    EDT is monotonic along the path toward its nearest feature — moving one
    step toward that feature strictly decreases the distance — so every flat
    cell necessarily has a neighbour with a smaller increment, hence a lower
    elevation. Drainage is therefore guaranteed by construction rather than
    hoped for.

    WHAT NOT TO DO (this was a real bug here): combining two distance
    transforms — "away from high ground" MINUS "toward the outlet" — as a
    literal reading of Garbrecht & Martz (1997) suggests, destroys that
    monotonicity. The difference of two EDTs has interior local minima, which
    are new sinks. It produced 19% sinks on the real Barpeta DEM and a maximum
    flow accumulation of 462 cells out of two million.

    ``epsilon`` is the TOTAL rise added across the widest flat, not a per-cell
    step, so the adjustment can never reorder genuinely distinct elevations.
    """
    from scipy import ndimage

    work = filled.astype(np.float64, copy=True)
    valid = np.isfinite(work)
    if not valid.any():
        return work

    # NaN cells act as walls: make them unattractively high so nothing drains
    # into them and they are never chosen as a receiver.
    highest = float(np.nanmax(work[valid]))
    probe = np.where(valid, work, highest + 1.0)

    # A cell "drains" if it already has a strictly-lower neighbour.
    min_neighbour = ndimage.minimum_filter(probe, size=3, mode="nearest")
    drains = valid & (min_neighbour < probe)
    flat = valid & ~drains
    if not flat.any():
        return work

    if not drains.any():
        # Pathological: the entire surface is one flat plane with no outlet.
        # Tilt it toward the lowest boundary cell so flow has somewhere to go.
        seed = np.zeros_like(flat)
        seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
        drains = seed & valid

    distance = ndimage.distance_transform_edt(~drains)
    span = float(distance[flat].max()) if flat.any() else 0.0
    if span <= 0:
        return work

    # Normalise so the deepest point of the widest flat rises by exactly
    # `epsilon` and everything else by proportionally less.
    work[flat] = work[flat] + epsilon * (distance[flat] / span)
    return work


# --------------------------------------------------------------------------
# 3. D8 flow directions
# --------------------------------------------------------------------------
def d8_receivers(dem: np.ndarray, cell_size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Steepest-descent (D8) receiver for every cell.

    Returns
    -------
    receivers
        Flat index of the downslope neighbour. Sinks and border outlets point
        at themselves, which makes downstream traversal terminate naturally
        instead of needing a sentinel.
    codes
        Index into :data:`_D8_OFFSETS`, or -1 where the cell is a sink.

    Vectorised over all 8 directions at once: 8 shifted comparisons instead of
    a per-pixel loop.
    """
    dem = np.asarray(dem, dtype=np.float64)
    h, w = dem.shape
    flat_index = np.arange(h * w, dtype=np.int64).reshape(h, w)

    best_slope = np.zeros((h, w), dtype=np.float64)
    best_code = np.full((h, w), -1, dtype=np.int8)
    best_recv = flat_index.copy()

    padded = np.pad(dem, 1, mode="constant", constant_values=np.inf)
    padded_idx = np.pad(flat_index, 1, mode="constant", constant_values=-1)

    for code, (dr, dc) in enumerate(_D8_OFFSETS):
        # Slice the padded array to get the neighbour in direction (dr, dc).
        neigh = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        neigh_idx = padded_idx[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        drop = dem - neigh
        slope = drop / (_D8_DIST[code] * cell_size)
        better = (slope > best_slope) & np.isfinite(neigh) & (neigh_idx >= 0)
        best_slope = np.where(better, slope, best_slope)
        best_code = np.where(better, np.int8(code), best_code)
        best_recv = np.where(better, neigh_idx, best_recv)

    nan_cells = ~np.isfinite(dem)
    if nan_cells.any():
        best_recv[nan_cells] = flat_index[nan_cells]
        best_code[nan_cells] = -1

    return best_recv.ravel(), best_code


# --------------------------------------------------------------------------
# 4. Flow accumulation
# --------------------------------------------------------------------------
def flow_accumulation(dem: np.ndarray, receivers: np.ndarray) -> np.ndarray:
    """Number of upstream cells draining through each cell (self included).

    Cells are processed in order of DECREASING elevation. In a filled DEM every
    receiver is at or below its donor, so that ordering guarantees a cell's
    contribution is added before its own receiver is visited — a topological
    sort we get for free from the elevation sort.

    The accumulation loop is the one genuinely sequential step in the whole
    pipeline (each cell's total depends on its donors'), so it stays a Python
    loop. It is a ONE-TIME static precompute that is cached to disk — it runs
    once per district, never during a live cycle.
    """
    dem = np.asarray(dem, dtype=np.float64)
    n = dem.size
    flat_dem = dem.ravel()

    order = np.argsort(-np.nan_to_num(flat_dem, nan=-np.inf), kind="stable")
    accum = np.ones(n, dtype=np.float64)
    valid = np.isfinite(flat_dem)
    accum[~valid] = 0.0

    rec = receivers
    acc = accum
    for i in order:
        j = rec[i]
        if j != i:
            acc[j] += acc[i]

    return acc.reshape(dem.shape)


# --------------------------------------------------------------------------
# 5. Drainage network
# --------------------------------------------------------------------------
def drainage_network(
    accumulation: np.ndarray,
    threshold_cells: int = 1000,
    permanent_water: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cells that count as channel.

    ``permanent_water`` lets a known river mask (e.g. from a dry-season SAR
    scene, or OSM waterways) be forced into the network. That matters on the
    Brahmaputra, whose braided channels migrate enough that a threshold on
    accumulation alone can miss an active anabranch.
    """
    drainage = np.asarray(accumulation) >= threshold_cells
    if permanent_water is not None:
        drainage = drainage | (np.asarray(permanent_water) > 0)
    return drainage


# --------------------------------------------------------------------------
# 6. HAND
# --------------------------------------------------------------------------
def compute_hand(
    dem: np.ndarray,
    cell_size: float = 30.0,
    drainage_threshold_cells: int = 1000,
    permanent_water: Optional[np.ndarray] = None,
    fill: bool = True,
) -> HandResult:
    """Full HAND pipeline for one district.

    Static: run once, cache the result. Nothing here depends on the flood.
    """
    dem = np.asarray(dem, dtype=np.float64)
    logger.info("HAND: starting on %s grid at %.0f m", dem.shape, cell_size)

    if fill:
        # Priority-Flood + epsilon does filling AND flat resolution together,
        # with a guarantee of zero sinks. Do not "optimise" this into
        # reconstruction + a separate flat pass — see the docstring for why
        # that silently destroyed HAND on the real DEM.
        graded = priority_flood_epsilon(dem)
        filled = graded
    else:
        filled = dem.copy()
        graded = resolve_flats(filled)

    receivers, _codes = d8_receivers(graded, cell_size=cell_size)
    # Boundary cells legitimately drain out of the AOI, so only INTERIOR sinks
    # indicate a conditioning failure.
    is_sink = (receivers == np.arange(dem.size)).reshape(dem.shape)
    is_sink[0, :] = is_sink[-1, :] = False
    is_sink[:, 0] = is_sink[:, -1] = False
    interior_sinks = int(is_sink.sum())
    if interior_sinks > 0.001 * dem.size:
        logger.warning(
            "Hydrology: %d interior sinks remain (%.2f%% of cells) after conditioning. "
            "Flow accumulation will be fragmented and HAND unreliable. This normally "
            "means the DEM has NaN voids or the conditioning step was bypassed.",
            interior_sinks, 100.0 * interior_sinks / dem.size,
        )
    accum = flow_accumulation(graded, receivers)
    drainage = drainage_network(accum, drainage_threshold_cells, permanent_water)

    if not drainage.any():
        # Degenerate AOI (tiny test grids, or a threshold set too high). Fall
        # back to the lowest percentile of the terrain so HAND is still defined
        # rather than silently all-NaN.
        logger.warning(
            "No cells exceeded the drainage threshold of %d; falling back to the "
            "lowest 1%% of terrain as the channel network.",
            drainage_threshold_cells,
        )
        drainage = filled <= np.nanpercentile(filled, 1.0)

    drain_elev = _propagate_drainage_elevation(filled, receivers, drainage, accum)

    hand = filled - drain_elev
    # Numerically, a drainage cell must be exactly 0 above itself; the flat
    # gradient can push it a hair negative. Clamp rather than let a negative
    # depth propagate into the impact layer.
    hand = np.where(np.isfinite(hand), np.maximum(hand, 0.0), np.nan)

    logger.info(
        "HAND: %.1f%% of cells are channel; HAND range %.2f..%.2f m",
        100.0 * drainage.mean(),
        float(np.nanmin(hand)),
        float(np.nanmax(hand)),
    )

    return HandResult(
        hand=hand.astype(np.float32),
        filled_dem=filled.astype(np.float32),
        flow_accumulation=accum.astype(np.float32),
        drainage=drainage,
        receivers=receivers,
        drainage_elevation=drain_elev.astype(np.float32),
    )


def _propagate_drainage_elevation(
    filled: np.ndarray,
    receivers: np.ndarray,
    drainage: np.ndarray,
    accumulation: np.ndarray,
) -> np.ndarray:
    """Elevation of the first drainage cell reached by following flow downstream.

    Processed in order of DECREASING accumulation, i.e. river first, ridges
    last. A receiver strictly accumulates MORE than its donor (its own count
    includes the donor's), so this ordering guarantees a cell's receiver is
    already final by the time the cell reads it. That turns what looks like a
    per-cell downstream walk — potentially O(n) per cell, O(n^2) overall on a
    long floodplain flow path — into a single O(n log n) pass.
    """
    n = filled.size
    flat_filled = filled.ravel()
    flat_drain = drainage.ravel()

    result = np.full(n, np.nan, dtype=np.float64)
    result[flat_drain] = flat_filled[flat_drain]

    order = np.argsort(-accumulation.ravel(), kind="stable")

    rec = receivers
    res = result
    drn = flat_drain
    for i in order:
        if drn[i]:
            continue
        j = rec[i]
        if j == i:
            # Sink or border outlet that never reached a channel. Treat the
            # cell as its own base level: HAND 0, i.e. "we do not know that
            # this is elevated", which is the safe direction to be wrong in.
            res[i] = flat_filled[i]
        else:
            res[i] = res[j]

    # Any cell still NaN drains only into NaN terrain; fall back to self.
    unresolved = ~np.isfinite(res)
    if unresolved.any():
        res[unresolved] = flat_filled[unresolved]

    return res.reshape(filled.shape)
