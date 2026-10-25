"""Road passability — which specific road segments are cut.

This is the join that makes the whole system concrete. "The district is
flooded" is a headline. "The Jania road is under 0.4 m of water between these
two junctions, and it is the only vehicle route out of this village" is a
decision.

Depth thresholds are configuration, not code, and every one of them is cited
in ``docs/REFERENCES.md``. The short version:

    < 0.15 m   CLEAR              normal driving
    < 0.30 m   CAUTION            passable on foot; unsafe for two-wheelers
    >= 0.30 m  VEHICLE_BLOCKED    small passenger vehicles lose stability
    >= 0.60 m  LIFE_THREATENING   do not send pedestrians either

Two modelling choices worth defending:

  * CONTROLLING DEPTH IS A HIGH PERCENTILE, NOT THE MEAN. A road is only as
    passable as its worst point — averaging a 300 m causeway with one flooded
    dip produces a comfortable number and a drowned driver. We use the 90th
    percentile of sampled depth rather than the maximum, so a single
    misclassified pixel cannot close a highway, but a genuinely flooded reach
    does.

  * CAUTION IS A SPEED PENALTY, NOT A CUT. Real evacuation through shallow
    water is slower, not impossible. Modelling it as a hard cut would route
    people the long way round for no reason; modelling it as free would
    under-estimate travel time. It is a cost multiplier.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..config import PassabilityConfig, EvacuationConfig
from ..grid import Grid

logger = logging.getLogger(__name__)

__all__ = [
    "PassabilityStatus",
    "annotate_road_passability",
    "RoadImpactSummary",
    "sample_edge_depths",
]


class PassabilityStatus(str, Enum):
    CLEAR = "clear"
    CAUTION = "caution"
    VEHICLE_BLOCKED = "vehicle_blocked"
    LIFE_THREATENING = "life_threatening"

    @property
    def vehicle_passable(self) -> bool:
        return self in (PassabilityStatus.CLEAR, PassabilityStatus.CAUTION)

    @property
    def foot_passable(self) -> bool:
        return self is not PassabilityStatus.LIFE_THREATENING

    @property
    def severity(self) -> int:
        return {
            PassabilityStatus.CLEAR: 0,
            PassabilityStatus.CAUTION: 1,
            PassabilityStatus.VEHICLE_BLOCKED: 2,
            PassabilityStatus.LIFE_THREATENING: 3,
        }[self]


@dataclass
class RoadImpactSummary:
    total_edges: int
    total_km: float
    clear_km: float
    caution_km: float
    vehicle_blocked_km: float
    life_threatening_km: float
    blocked_edges: int
    worst_depth_m: float

    @property
    def cut_fraction(self) -> float:
        return (self.vehicle_blocked_km + self.life_threatening_km) / self.total_km if self.total_km else 0.0

    def to_dict(self) -> dict:
        return {
            "total_edges": self.total_edges,
            "total_km": round(self.total_km, 2),
            "clear_km": round(self.clear_km, 2),
            "caution_km": round(self.caution_km, 2),
            "vehicle_blocked_km": round(self.vehicle_blocked_km, 2),
            "life_threatening_km": round(self.life_threatening_km, 2),
            "blocked_edges": self.blocked_edges,
            "cut_fraction": round(self.cut_fraction, 4),
            "worst_depth_m": round(self.worst_depth_m, 2),
        }


# --------------------------------------------------------------------------
# Geometry sampling
# --------------------------------------------------------------------------
def _edge_points(graph: Any, u: Any, v: Any, data: dict, spacing: float) -> np.ndarray:
    """Densified sample points along one edge, in grid CRS.

    Uses the true OSM geometry when present. Falls back to the straight chord
    between endpoints otherwise — which for a curved rural road can miss the
    flooded bend entirely, so we prefer real geometry whenever osmnx gives it.
    """
    coords: Optional[Sequence[tuple[float, float]]] = None
    geom = data.get("geometry")

    if geom is not None:
        if hasattr(geom, "coords"):                 # shapely LineString
            coords = list(geom.coords)
        elif isinstance(geom, (list, tuple)) and geom:
            coords = [(float(p[0]), float(p[1])) for p in geom]

    if not coords:
        nu, nv = graph.nodes[u], graph.nodes[v]
        coords = [(float(nu["x"]), float(nu["y"])), (float(nv["x"]), float(nv["y"]))]

    pts: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n = max(int(seg_len / spacing), 1)
        for i in range(n):
            t = i / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    pts.append((float(coords[-1][0]), float(coords[-1][1])))
    return np.asarray(pts, dtype=np.float64)


def sample_edge_depths(
    graph: Any,
    grid: Grid,
    depth: np.ndarray,
    spacing: Optional[float] = None,
    percentile: float = 90.0,
) -> dict[tuple[Any, Any, Any], tuple[float, float]]:
    """Sample flood depth along every edge.

    Returns ``{(u, v, key): (controlling_depth, max_depth)}``. Sampling
    spacing defaults to half a pixel so no cell along the edge is skipped —
    sampling at exactly the pixel size aliases and can step straight over a
    one-cell-wide flooded channel.
    """
    spacing = spacing or grid.resolution / 2.0
    out: dict[tuple[Any, Any, Any], tuple[float, float]] = {}

    for u, v, k, data in graph.edges(keys=True, data=True):
        pts = _edge_points(graph, u, v, data, spacing)
        values = grid.sample(depth, pts[:, 0], pts[:, 1], default=0.0)
        values = np.nan_to_num(values, nan=0.0)
        if values.size == 0:
            out[(u, v, k)] = (0.0, 0.0)
            continue
        out[(u, v, k)] = (float(np.percentile(values, percentile)), float(values.max()))
    return out


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def classify_depth(depth_m: float, cfg: PassabilityConfig) -> PassabilityStatus:
    if depth_m >= cfg.life_threatening_m:
        return PassabilityStatus.LIFE_THREATENING
    if depth_m >= cfg.vehicle_block_m:
        return PassabilityStatus.VEHICLE_BLOCKED
    if depth_m >= cfg.clear_m:
        return PassabilityStatus.CAUTION
    return PassabilityStatus.CLEAR


def annotate_road_passability(
    graph: Any,
    grid: Grid,
    depth: np.ndarray,
    passability: PassabilityConfig,
    evacuation: EvacuationConfig,
    percentile: float = 90.0,
) -> RoadImpactSummary:
    """Write depth, status and travel cost onto every edge, in place.

    Edge attributes set:
        ``flood_depth_m``       controlling depth (percentile)
        ``flood_depth_max_m``   worst sampled depth
        ``passability``         :class:`PassabilityStatus` value
        ``travel_time_s``       vehicle traversal time under flood conditions
        ``walk_time_s``         pedestrian traversal time under flood conditions

    Both a vehicle and a walking cost are produced because the two evacuation
    modes have different blocking thresholds. A road that stops a car does not
    necessarily stop a person, and on foot is how most of rural Assam actually
    evacuates.
    """
    depths = sample_edge_depths(graph, grid, depth, percentile=percentile)

    veh_mps = evacuation.vehicle_speed_kmph * 1000.0 / 3600.0
    walk_mps = evacuation.walking_speed_kmph * 1000.0 / 3600.0

    km = {s: 0.0 for s in PassabilityStatus}
    blocked_edges = 0
    worst = 0.0
    total_km = 0.0

    for u, v, k, data in graph.edges(keys=True, data=True):
        controlling, dmax = depths.get((u, v, k), (0.0, 0.0))
        status = classify_depth(controlling, passability)
        length_m = float(data.get("length", 0.0) or 0.0)
        if length_m <= 0:
            nu, nv = graph.nodes[u], graph.nodes[v]
            length_m = math.hypot(float(nu["x"]) - float(nv["x"]), float(nu["y"]) - float(nv["y"]))
            data["length"] = length_m

        data["flood_depth_m"] = round(controlling, 3)
        data["flood_depth_max_m"] = round(dmax, 3)
        data["passability"] = status.value

        # Vehicle cost
        if status is PassabilityStatus.CLEAR:
            data["travel_time_s"] = length_m / veh_mps
        elif status is PassabilityStatus.CAUTION:
            data["travel_time_s"] = length_m / (veh_mps * passability.caution_speed_factor)
        else:
            data["travel_time_s"] = math.inf

        # Pedestrian cost — blocked only by the life-threatening band.
        if status is PassabilityStatus.LIFE_THREATENING:
            data["walk_time_s"] = math.inf
        elif status is PassabilityStatus.CLEAR:
            data["walk_time_s"] = length_m / walk_mps
        else:
            data["walk_time_s"] = length_m / (walk_mps * passability.caution_speed_factor)

        km[status] += length_m / 1000.0
        total_km += length_m / 1000.0
        worst = max(worst, dmax)
        if not status.vehicle_passable:
            blocked_edges += 1

    summary = RoadImpactSummary(
        total_edges=graph.number_of_edges(),
        total_km=total_km,
        clear_km=km[PassabilityStatus.CLEAR],
        caution_km=km[PassabilityStatus.CAUTION],
        vehicle_blocked_km=km[PassabilityStatus.VEHICLE_BLOCKED],
        life_threatening_km=km[PassabilityStatus.LIFE_THREATENING],
        blocked_edges=blocked_edges,
        worst_depth_m=worst,
    )
    logger.info(
        "Roads: %.1f km of %.1f km cut to vehicles (%.1f%%), worst depth %.2f m",
        summary.vehicle_blocked_km + summary.life_threatening_km,
        summary.total_km, 100 * summary.cut_fraction, summary.worst_depth_m,
    )
    return summary
