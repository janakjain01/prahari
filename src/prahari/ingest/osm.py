"""OpenStreetMap — the road network and the shelter candidates.

Two fetches, and each has a real caveat worth stating rather than glossing.

ROADS. osmnx returns a routable graph directly, which is why this module is
short. We project it into the district CRS immediately: routing on lat/lon
gives edge "lengths" in degrees, and a degree is not a distance.

SHELTERS. Assam relief camps are overwhelmingly schools, so OSM
``amenity=school`` is a good candidate list. But OSM gives us a POINT or a
building outline, never a CAPACITY. Occupancy is therefore ESTIMATED from
footprint area, and every downstream surface says so. The right fix is the
district relief-camp register, which is not open data — that is a real
deployment dependency, and naming it is more credible than inventing numbers.

OSM completeness in rural Assam is uneven. Some villages have every lane
mapped and some have one line through the middle. This affects the priority
list directly: a settlement can look "cut off" because its road was never
mapped. That is why unbound settlements get their OWN reason code
(``UNBOUND``) instead of being silently folded into ``CUT_OFF``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..config import DistrictConfig
from ..grid import Grid
from ..scenario import Shelter
from .base import IngestResult, cache_path, requires, retry

logger = logging.getLogger(__name__)

__all__ = ["fetch_road_network", "fetch_shelter_candidates", "fetch_embankments"]


@requires("osmnx")
@retry(attempts=2, backoff_s=4.0)
def fetch_road_network(
    bbox: tuple[float, float, float, float],
    grid: Grid,
    network_type: str = "drive",
    district_id: str = "unknown",
    use_cache: bool = True,
) -> IngestResult[Any]:
    """Download a routable road graph and project it into the district CRS."""
    import networkx as nx
    import osmnx as ox

    key = f"roads|{network_type}|{bbox}"
    cached = cache_path(district_id, "osm", key, suffix=".graphml")

    if use_cache and cached.exists():
        try:
            graph = ox.load_graphml(cached)
            graph = _to_grid_crs(graph, grid, ox)
            return IngestResult(
                ok=True, data=graph, source="openstreetmap", from_cache=True,
                detail=f"cached {network_type} network, {graph.number_of_edges()} edges",
            )
        except Exception as exc:
            logger.warning("OSM cache unreadable (%s); refetching", exc)

    west, south, east, north = bbox
    # osmnx 2.x takes (left, bottom, right, top); 1.x took (north, south, east, west).
    try:
        graph = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)
    except TypeError:
        graph = ox.graph_from_bbox(north, south, east, west, network_type=network_type)

    if graph.number_of_edges() == 0:
        return IngestResult.failed("openstreetmap", f"no {network_type} roads mapped in {bbox}")

    if use_cache:
        try:
            ox.save_graphml(graph, cached)
        except Exception as exc:
            logger.warning("Could not cache OSM graph: %s", exc)

    graph = _to_grid_crs(graph, grid, ox)
    logger.info(
        "OSM: %d nodes / %d edges (%s network)",
        graph.number_of_nodes(), graph.number_of_edges(), network_type,
    )
    return IngestResult(
        ok=True, data=graph, source="openstreetmap",
        detail=f"{network_type} network, {graph.number_of_edges()} edges",
        meta={"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
    )


def _to_grid_crs(graph, grid: Grid, ox) -> Any:
    """Project the graph and normalise node coordinates to ``x``/``y`` in metres.

    Everything downstream — edge sampling, KD-tree binding, travel time —
    assumes metric node coordinates on the district grid. Doing this once here
    means no other module has to think about projection.
    """
    projected = ox.project_graph(graph, to_crs=grid.crs)
    for _n, data in projected.nodes(data=True):
        if "x" not in data and "lon" in data:
            data["x"] = data["lon"]
        if "y" not in data and "lat" in data:
            data["y"] = data["lat"]
    return projected


@requires("osmnx", "geopandas")
@retry(attempts=2, backoff_s=4.0)
def fetch_shelter_candidates(
    bbox: tuple[float, float, float, float],
    grid: Grid,
    config: DistrictConfig,
    district_id: str = "unknown",
    min_footprint_sqm: float = 120.0,
) -> IngestResult[list[Shelter]]:
    """Harvest schools, colleges and community halls as shelter candidates."""
    import osmnx as ox

    from ..impact.shelters import estimate_capacity_from_footprint

    tags = config.sources.osm.get("shelter_tags") or {
        "amenity": ["school", "college", "community_centre", "hospital", "shelter"]
    }

    west, south, east, north = bbox
    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    except TypeError:
        gdf = ox.features_from_bbox(north, south, east, west, tags=tags)
    except Exception as exc:
        return IngestResult.failed("openstreetmap", f"shelter query failed: {exc}")

    if gdf is None or len(gdf) == 0:
        return IngestResult.failed("openstreetmap", f"no shelter-type features mapped in {bbox}")

    gdf = gdf.to_crs(grid.crs)
    shelters: list[Shelter] = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Footprint area drives the capacity estimate. A node has no area, so
        # it falls back to the configured default and is labelled as such.
        footprint = float(geom.area) if geom.geom_type in ("Polygon", "MultiPolygon") else 0.0
        if 0 < footprint < min_footprint_sqm:
            continue

        point = geom.centroid
        kind = str(row.get("amenity") or row.get("building") or "shelter")
        name = str(row.get("name") or f"Unnamed {kind.replace('_', ' ')}")

        try:
            if footprint > 0:
                capacity = estimate_capacity_from_footprint(
                    footprint, config.evacuation.sqm_per_person, storeys=_storeys(row)
                )
                source = "footprint_estimate"
            else:
                capacity = config.evacuation.default_shelter_capacity
                source = "default_assumption"
        except (TypeError, ValueError) as exc:
            # One malformed OSM tag must not take down the whole harvest.
            logger.debug("Skipping shelter %s: %s", idx, exc)
            continue

        if not np.isfinite(capacity) or capacity < 40:
            continue

        shelters.append(
            Shelter(
                id=f"osm{_osm_id(idx)}",
                name=name,
                x=float(point.x), y=float(point.y),
                capacity=int(capacity),
                capacity_source=source,
                kind=kind,
                footprint_sqm=footprint or None,
            )
        )

    if not shelters:
        return IngestResult.failed("openstreetmap", "shelter features found but none usable")

    estimated = sum(1 for s in shelters if s.capacity_source != "surveyed")
    logger.info(
        "OSM: %d shelter candidates, total nominal capacity %d — ALL capacities are "
        "estimated (%d/%d), none surveyed",
        len(shelters), sum(s.capacity for s in shelters), estimated, len(shelters),
    )
    return IngestResult(
        ok=True, data=shelters, source="openstreetmap",
        detail=f"{len(shelters)} candidates; capacity estimated from footprint, NOT surveyed",
        meta={"count": len(shelters), "capacity": sum(s.capacity for s in shelters)},
    )


def _storeys(row) -> float:
    """Building storeys from OSM tags, defaulting to 1.

    OSM tag values arrive through geopandas as object columns, so a missing
    value is float('nan') rather than None. ``float('nan')`` parses happily,
    ``max(nan, 1.0)`` returns nan, and ``int(nan)`` then raises — which is how
    one unmapped building used to abort the whole shelter harvest for a
    district. Hence the explicit non-finite check.
    """
    for field in ("building:levels", "levels"):
        value = row.get(field)
        if value is None:
            continue
        try:
            parsed = float(str(value).split(";")[0].strip())
        except (TypeError, ValueError):
            continue
        if not np.isfinite(parsed) or parsed <= 0:
            continue
        return max(parsed, 1.0)
    return 1.0


def _osm_id(idx) -> str:
    if isinstance(idx, tuple):
        return "_".join(str(part) for part in idx)
    return str(idx)


@requires("osmnx", "geopandas")
@retry(attempts=2, backoff_s=4.0)
def fetch_place_names(
    bbox: tuple[float, float, float, float],
    grid: Grid,
) -> IngestResult[list[dict]]:
    """Named populated places from OSM, for labelling derived settlements.

    Settlements are clustered from the population raster, so they come out as
    "Settlement 108". That is useless operationally — a dispatch list has to
    say a name a district officer and a boat crew both recognise. This fetches
    OSM ``place=`` nodes so clusters can be given the name of the village they
    actually sit on.

    Returns ``[{name, kind, x, y}]`` in grid CRS, ordered from largest place
    type to smallest so a city name wins over a neighbouring hamlet when both
    are nearby.
    """
    import osmnx as ox

    tags = {"place": ["city", "town", "village", "hamlet", "suburb", "neighbourhood"]}
    west, south, east, north = bbox
    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    except TypeError:
        gdf = ox.features_from_bbox(north, south, east, west, tags=tags)
    except Exception as exc:
        return IngestResult.failed("openstreetmap", f"place query failed: {exc}")

    if gdf is None or len(gdf) == 0:
        return IngestResult.failed("openstreetmap", f"no named places mapped in {bbox}")

    gdf = gdf.to_crs(grid.crs)
    rank = {"city": 0, "town": 1, "suburb": 2, "village": 3, "neighbourhood": 4, "hamlet": 5}

    places: list[dict] = []
    for _idx, row in gdf.iterrows():
        name = row.get("name")
        geom = row.geometry
        if not name or geom is None or geom.is_empty:
            continue
        point = geom.centroid
        kind = str(row.get("place") or "village")
        places.append(
            {"name": str(name), "kind": kind, "x": float(point.x), "y": float(point.y),
             "rank": rank.get(kind, 9)}
        )

    if not places:
        return IngestResult.failed("openstreetmap", "place features found but none are named")

    places.sort(key=lambda p: p["rank"])
    logger.info("OSM: %d named places for settlement labelling", len(places))
    return IngestResult(
        ok=True, data=places, source="openstreetmap",
        detail=f"{len(places)} named places",
    )


_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _bearing_label(dx: float, dy: float) -> str:
    """Eight-point compass direction from a metric offset."""
    import math

    angle = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return _COMPASS[int((angle + 22.5) // 45) % 8]


def label_settlements(
    settlements,
    places: list[dict],
    exact_distance_m: float = 2500.0,
    relative_distance_m: float = 12000.0,
) -> dict:
    """Give derived settlement clusters names a boat crew can act on.

    Two tiers, because OSM coverage of rural Assam is thin — the whole of
    Barpeta district has about a dozen mapped place nodes for hundreds of real
    villages:

      EXACT     a cluster within ``exact_distance_m`` of a named place takes
                that name. Each name is used AT MOST ONCE — two clusters both
                called "Jania" is an operational hazard, not a cosmetic one,
                because two different villages that read identically get
                conflated over the radio.

      RELATIVE  everything else is labelled by bearing and distance from the
                nearest named place: "4 km SW of Mandia". That is derived, not
                an official name, and it is phrased so nobody could mistake it
                for one — but it is something a crew can actually navigate to,
                which "Settlement 108" is not.

    Mutates ``settlements`` in place. Returns a small report.
    """
    import math

    if not places or not settlements:
        return {"exact": 0, "relative": 0, "generic": len(settlements)}

    taken: set[str] = set()
    assigned: set[int] = set()

    pairs: list[tuple[float, int, int]] = []
    for si, s in enumerate(settlements):
        for pi, p in enumerate(places):
            distance = math.hypot(s.x - p["x"], s.y - p["y"])
            if distance <= exact_distance_m:
                pairs.append((distance, si, pi))
    pairs.sort()

    exact = 0
    for _distance, si, pi in pairs:
        name = places[pi]["name"]
        if si in assigned or name in taken:
            continue
        settlements[si].name = name
        assigned.add(si)
        taken.add(name)
        exact += 1

    relative = 0
    for si, s in enumerate(settlements):
        if si in assigned:
            continue
        nearest = min(places, key=lambda p: math.hypot(s.x - p["x"], s.y - p["y"]))
        dx, dy = s.x - nearest["x"], s.y - nearest["y"]
        distance = math.hypot(dx, dy)
        if distance > relative_distance_m:
            continue
        settlements[si].name = (
            f"{distance / 1000:.0f} km {_bearing_label(dx, dy)} of {nearest['name']}"
        )
        relative += 1

    generic = len(settlements) - exact - relative
    logger.info(
        "Settlement names: %d exact from OSM, %d relative bearings, %d left generic "
        "(OSM has only %d named places for this district)",
        exact, relative, generic, len(places),
    )
    return {"exact": exact, "relative": relative, "generic": generic, "places": len(places)}


@requires("osmnx", "geopandas")
def fetch_embankments(
    bbox: tuple[float, float, float, float],
    grid: Grid,
) -> IngestResult[Any]:
    """Embankment and dyke centrelines, for burning into the DEM.

    A 30 m DEM cannot resolve these structures, and in Assam they are the
    thing standing between the river and the district. See
    ``dem.burn_embankments``.
    """
    import osmnx as ox

    tags = {"man_made": ["dyke", "embankment"], "embankment": True, "barrier": ["embankment"]}
    west, south, east, north = bbox
    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    except TypeError:
        gdf = ox.features_from_bbox(north, south, east, west, tags=tags)
    except Exception as exc:
        return IngestResult.failed("openstreetmap", f"embankment query failed: {exc}")

    if gdf is None or len(gdf) == 0:
        return IngestResult.failed(
            "openstreetmap",
            "no embankments mapped here. They almost certainly exist on the ground — "
            "OSM coverage of flood infrastructure in rural Assam is sparse. The DEM "
            "will therefore under-represent flood defences.",
        )

    lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(grid.crs)
    if len(lines) == 0:
        return IngestResult.failed("openstreetmap", "embankment features present but none are lines")

    logger.info("OSM: %d embankment centrelines", len(lines))
    return IngestResult(
        ok=True, data=lines, source="openstreetmap",
        detail=f"{len(lines)} embankment/dyke centrelines",
    )
