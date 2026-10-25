"""Safe-route computation on the pruned graph.

Shortest path is the easy half. The half that makes the output usable is the
COUNTERFACTUAL: we also compute the route an ordinary navigation app would
give — shortest path on the UNPRUNED network — and compare.

Where the two differ, we can name the specific road we are steering people
away from, and why::

    "Take the road via Jania — NOT the main road, it is under water."

That sentence is only possible because we routed twice. It is also the entire
pitch in one line, and it is generated from data rather than written into a
slide by hand:

    "Google Maps will route you straight into the water. We route you around
     it — and we make sure the shelter we send you to still has room."
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from ..grid import Grid
from ..scenario import Settlement, Shelter
from .assignment import Assignment
from .graph import TravelMode

logger = logging.getLogger(__name__)

__all__ = ["Route", "compute_routes", "route_geometry"]


@dataclass
class Route:
    """One evacuation route, ready to draw and ready to read aloud."""

    settlement_id: str
    settlement_name: str
    shelter_id: str
    shelter_name: str
    people: int
    mode: str
    distance_m: float
    travel_seconds: float
    coordinates: list[tuple[float, float]] = field(default_factory=list)   # (lon, lat)
    road_names: list[str] = field(default_factory=list)
    avoided_roads: list[str] = field(default_factory=list)
    naive_route_would_flood: bool = False
    naive_route_max_depth_m: float = 0.0
    detour_factor: float = 1.0
    max_depth_on_route_m: float = 0.0

    @property
    def travel_minutes(self) -> float:
        return self.travel_seconds / 60.0

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000.0

    def to_dict(self) -> dict:
        return {
            "settlement_id": self.settlement_id,
            "settlement_name": self.settlement_name,
            "shelter_id": self.shelter_id,
            "shelter_name": self.shelter_name,
            "people": self.people,
            "mode": self.mode,
            "distance_km": round(self.distance_km, 2),
            "travel_minutes": round(self.travel_minutes, 1),
            "road_names": self.road_names[:8],
            "avoided_roads": self.avoided_roads[:5],
            "naive_route_would_flood": self.naive_route_would_flood,
            "naive_route_max_depth_m": round(self.naive_route_max_depth_m, 2),
            "detour_factor": round(self.detour_factor, 2),
            "max_depth_on_route_m": round(self.max_depth_on_route_m, 2),
        }

    def to_geojson(self) -> dict:
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [list(c) for c in self.coordinates]},
            "properties": self.to_dict(),
        }


def compute_routes(
    assignments: Sequence[Assignment],
    pruned_graph: Any,
    original_graph: Any,
    settlements: Sequence[Settlement],
    shelters: Sequence[Shelter],
    grid: Grid,
    mode: TravelMode = TravelMode.VEHICLE,
    compute_counterfactual: bool = True,
    max_routes: Optional[int] = None,
) -> list[Route]:
    """Turn assignments into drawable, speakable routes.

    ``max_routes`` caps the work for interactive refresh; assignments arrive
    sorted by people, so the cap keeps the largest movements. It does not
    change the plan, only how much of it we draw this cycle.
    """
    import networkx as nx

    weight = mode.weight_attribute
    settlement_map = {s.id: s for s in settlements}
    shelter_map = {s.id: s for s in shelters}

    working = pruned_graph.to_undirected(as_view=False) if mode is TravelMode.FOOT else pruned_graph
    naive = original_graph.to_undirected(as_view=False) if mode is TravelMode.FOOT else original_graph

    routes: list[Route] = []
    failures = 0

    for assignment in (assignments[:max_routes] if max_routes else assignments):
        settlement = settlement_map.get(assignment.settlement_id)
        shelter = shelter_map.get(assignment.shelter_id)
        if settlement is None or shelter is None:
            continue
        if settlement.node is None or shelter.node is None:
            continue

        try:
            _cost, path = nx.single_source_dijkstra(
                working, settlement.node, shelter.node, weight=weight
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            failures += 1
            continue

        coords, names, distance, seconds, max_depth = _walk_path(working, path, grid, weight)

        route = Route(
            settlement_id=settlement.id,
            settlement_name=settlement.name,
            shelter_id=shelter.id,
            shelter_name=shelter.name,
            people=assignment.people,
            mode=mode.value,
            distance_m=distance,
            travel_seconds=seconds,
            coordinates=coords,
            road_names=names,
            max_depth_on_route_m=max_depth,
        )

        if compute_counterfactual:
            _attach_counterfactual(route, naive, settlement.node, shelter.node, distance)

        routes.append(route)

    if failures:
        logger.warning(
            "%d assignment(s) had no path on the pruned graph despite a finite travel time. "
            "This normally means the graph changed between matrix construction and routing.",
            failures,
        )
    steered = sum(1 for r in routes if r.naive_route_would_flood)
    logger.info(
        "Routes: %d computed; %d steered away from a flooded road a naive router would have used",
        len(routes), steered,
    )
    return routes


def _walk_path(
    graph: Any, path: Sequence[Any], grid: Grid, weight: str
) -> tuple[list[tuple[float, float]], list[str], float, float, float]:
    """Collect geometry, road names, length, time and worst depth along a path."""
    coords_xy: list[tuple[float, float]] = []
    names: list[str] = []
    distance = 0.0
    seconds = 0.0
    max_depth = 0.0

    for u, v in zip(path[:-1], path[1:]):
        data = _cheapest_edge(graph, u, v, weight)
        if data is None:
            continue
        distance += float(data.get("length", 0.0) or 0.0)
        cost = data.get(weight, 0.0)
        seconds += float(cost) if math.isfinite(float(cost)) else 0.0
        max_depth = max(max_depth, float(data.get("flood_depth_m", 0.0) or 0.0))

        name = _road_name(data)
        if name and (not names or names[-1] != name):
            names.append(name)

        coords_xy.extend(_edge_coords(graph, u, v, data))

    if not coords_xy and path:
        coords_xy = [
            (float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"]))
            for n in path
            if "x" in graph.nodes[n]
        ]

    coords_xy = _dedupe(coords_xy)
    if coords_xy:
        xs = [c[0] for c in coords_xy]
        ys = [c[1] for c in coords_xy]
        lon, lat = grid.to_geographic(xs, ys)
        coords = [(round(float(a), 6), round(float(b), 6)) for a, b in zip(lon, lat)]
    else:
        coords = []

    return coords, names, distance, seconds, max_depth


def _cheapest_edge(graph: Any, u: Any, v: Any, weight: str) -> Optional[dict]:
    """Pick the parallel edge Dijkstra would have used.

    OSM road networks are multigraphs — two junctions can be joined by both a
    100 m link and a 900 m loop. Taking an arbitrary parallel edge would report
    a length that does not match the route actually chosen.
    """
    try:
        bundle = graph[u][v]
    except KeyError:
        return None
    best, best_cost = None, math.inf
    for _key, data in bundle.items():
        cost = float(data.get(weight, data.get("length", math.inf)) or math.inf)
        if cost < best_cost:
            best, best_cost = data, cost
    return best


def _edge_coords(graph: Any, u: Any, v: Any, data: dict) -> list[tuple[float, float]]:
    geom = data.get("geometry")
    if geom is not None:
        if hasattr(geom, "coords"):
            return [(float(x), float(y)) for x, y in geom.coords]
        if isinstance(geom, (list, tuple)) and geom:
            return [(float(p[0]), float(p[1])) for p in geom]
    out = []
    for n in (u, v):
        nd = graph.nodes.get(n, {})
        if "x" in nd and "y" in nd:
            out.append((float(nd["x"]), float(nd["y"])))
    return out


def _road_name(data: dict) -> str:
    name = data.get("name") or data.get("ref")
    if isinstance(name, (list, tuple)):
        name = name[0] if name else None
    if not name:
        highway = data.get("highway")
        if isinstance(highway, (list, tuple)):
            highway = highway[0] if highway else None
        return f"unnamed {highway}" if highway else ""
    return str(name)


def _dedupe(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        if not out or abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out


def _attach_counterfactual(
    route: Route, naive_graph: Any, source: Any, target: Any, safe_distance_m: float
) -> None:
    """Route again on the UNPRUNED graph and record what we steered away from.

    Costed by ``length``, not by flood-aware travel time — that is precisely
    the point. This reproduces what a router with no flood information would
    return, which is what people's phones will actually tell them.
    """
    import networkx as nx

    try:
        _cost, naive_path = nx.single_source_dijkstra(
            naive_graph, source, target, weight="length"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return

    flooded_names: list[str] = []
    worst = 0.0
    naive_distance = 0.0

    for u, v in zip(naive_path[:-1], naive_path[1:]):
        data = _cheapest_edge(naive_graph, u, v, "length")
        if data is None:
            continue
        naive_distance += float(data.get("length", 0.0) or 0.0)
        depth = float(data.get("flood_depth_m", 0.0) or 0.0)
        status = str(data.get("passability", "clear"))
        if status in ("vehicle_blocked", "life_threatening"):
            worst = max(worst, depth)
            name = _road_name(data)
            if name and name not in flooded_names:
                flooded_names.append(name)

    route.naive_route_would_flood = bool(flooded_names)
    route.naive_route_max_depth_m = worst
    route.avoided_roads = flooded_names
    if naive_distance > 0:
        route.detour_factor = safe_distance_m / naive_distance


def route_geometry(routes: Sequence[Route]) -> dict:
    """All routes as one GeoJSON FeatureCollection, for the map layer."""
    return {"type": "FeatureCollection", "features": [r.to_geojson() for r in routes]}
