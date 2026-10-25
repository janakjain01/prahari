"""The Scenario — one district, one moment in time, every layer on one grid.

A ``Scenario`` is the single object that flows through the whole pipeline:

    ingest -> Scenario(static layers)
    hazard -> + flood_mask, depth
    impact -> + road passability, exposure
    decision -> EvacuationPlan

Splitting STATIC layers (terrain, roads, people, shelters — fetched once) from
DYNAMIC layers (flood extent and depth — recomputed every cycle) is what lets
the replay harness step an event forward hour by hour without re-ingesting
anything. It is also what makes the live demo fast enough to be worth watching.

Every layer carries provenance. When a judge asks "where did the population
figure come from?", the answer is in the object, not in someone's memory.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import numpy as np

from .grid import Grid

__all__ = [
    "Provenance",
    "Shelter",
    "Settlement",
    "Scenario",
    "CapacitySource",
]

CapacitySource = Literal["surveyed", "footprint_estimate", "default_assumption"]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
@dataclass
class Provenance:
    """Where a layer came from. Surfaced in the UI, not just logged.

    Honesty about data origin is a scoring asset, not a liability. A dashboard
    that says "shelter capacity: ESTIMATED from building footprint" is more
    credible than one that shows a confident number it cannot justify.
    """

    layer: str
    source: str
    detail: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_estimate: bool = False
    is_synthetic: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Point features
# --------------------------------------------------------------------------
@dataclass
class Shelter:
    """A candidate evacuation destination.

    ``capacity`` is the headline number but ``capacity_source`` is the honest
    one. We never silently present an estimate as a survey.
    """

    id: str
    name: str
    x: float
    y: float
    capacity: int
    capacity_source: CapacitySource = "default_assumption"
    kind: str = "school"
    footprint_sqm: Optional[float] = None
    node: Optional[int] = None          # nearest road-graph node, set at bind time
    elevation_m: Optional[float] = None
    # Filled by the impact layer: a shelter standing in floodwater is not a
    # shelter, however large its hall.
    flood_depth_m: float = 0.0
    is_viable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Shelter:
        return cls(**d)


@dataclass
class Settlement:
    """A populated place that may need to move.

    Derived from population raster clustering, or from OSM place nodes when
    available. Carries the evacuation demand.
    """

    id: str
    name: str
    x: float
    y: float
    population: float
    node: Optional[int] = None
    # Filled by the impact layer.
    flood_depth_m: float = 0.0
    is_flooded: bool = False
    # People who actually need to move. This is NOT the same as ``population``
    # and conflating the two is a modelling error: you do not evacuate a
    # district, you evacuate the part of it that is under water or about to
    # be. Set by impact.population.compute_evacuation_demand().
    people_at_risk: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Settlement:
        return cls(**d)


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    """All layers for one district at one timestamp."""

    district_id: str
    grid: Grid

    # --- static layers (metres / people per pixel) ---
    dem: np.ndarray                       # elevation, float32
    hand: np.ndarray                      # height above nearest drainage, float32
    population: np.ndarray                # people per pixel, float32
    landcover: Optional[np.ndarray] = None

    # --- dynamic layers ---
    flood_mask: Optional[np.ndarray] = None    # uint8 {0,1}
    depth: Optional[np.ndarray] = None         # metres, float32
    confidence: Optional[np.ndarray] = None    # 0..1, float32

    # --- vector layers ---
    roads: Any = None                     # networkx.MultiDiGraph
    shelters: list[Shelter] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)

    # --- metadata ---
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: list[Provenance] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- validation --------------------------------------------------------
    def __post_init__(self) -> None:
        for name in ("dem", "hand", "population"):
            arr = getattr(self, name)
            if arr is None:
                continue
            if arr.shape != self.grid.shape:
                raise ValueError(
                    f"layer '{name}' has shape {arr.shape} but grid is {self.grid.shape}. "
                    "All layers must share one grid — resample at ingest time."
                )

    # -- convenience -------------------------------------------------------
    @property
    def is_synthetic(self) -> bool:
        return any(p.is_synthetic for p in self.provenance)

    @property
    def has_hazard(self) -> bool:
        return self.flood_mask is not None and self.depth is not None

    def add_provenance(self, layer: str, source: str, **kw: Any) -> None:
        self.provenance.append(Provenance(layer=layer, source=source, **kw))

    def provenance_for(self, layer: str) -> Optional[Provenance]:
        for p in reversed(self.provenance):
            if p.layer == layer:
                return p
        return None

    def total_population(self) -> float:
        return float(np.nansum(self.population))

    def flooded_area_km2(self) -> float:
        if self.flood_mask is None:
            return 0.0
        return float(self.flood_mask.sum()) * self.grid.cell_area_m2 / 1e6

    def exposed_population(self) -> float:
        """People standing in water right now. The headline number."""
        if self.flood_mask is None:
            return 0.0
        return float(np.nansum(self.population * (self.flood_mask > 0)))

    def viable_shelters(self) -> list[Shelter]:
        return [s for s in self.shelters if s.is_viable]

    def total_shelter_capacity(self, usable_factor: float = 1.0) -> int:
        return int(sum(s.capacity for s in self.viable_shelters()) * usable_factor)

    def shelter_by_id(self, sid: str) -> Optional[Shelter]:
        return next((s for s in self.shelters if s.id == sid), None)

    def settlement_by_id(self, sid: str) -> Optional[Settlement]:
        return next((s for s in self.settlements if s.id == sid), None)

    def summary(self) -> dict[str, Any]:
        """Compact status block — drives the dashboard header and the CLI."""
        return {
            "district_id": self.district_id,
            "timestamp": self.timestamp,
            "synthetic": self.is_synthetic,
            "grid": {
                "shape": list(self.grid.shape),
                "resolution_m": self.grid.resolution,
                "crs": self.grid.crs,
                "area_km2": round(self.grid.size * self.grid.cell_area_m2 / 1e6, 1),
            },
            "population_total": round(self.total_population()),
            "settlements": len(self.settlements),
            "shelters_total": len(self.shelters),
            "shelters_viable": len(self.viable_shelters()),
            "shelter_capacity": self.total_shelter_capacity(),
            "roads_edges": self.roads.number_of_edges() if self.roads is not None else 0,
            "roads_nodes": self.roads.number_of_nodes() if self.roads is not None else 0,
            "hazard": {
                "present": self.has_hazard,
                "flooded_km2": round(self.flooded_area_km2(), 1),
                "exposed_population": round(self.exposed_population()),
            },
            # Present only for real satellite observations; the dashboard uses
            # it to state the acquisition time and swath coverage on screen.
            "observation": self.meta.get("observation"),
        }

    # -- copying -----------------------------------------------------------
    def with_hazard(
        self,
        flood_mask: np.ndarray,
        depth: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        timestamp: Optional[str] = None,
        provenance: Optional[Provenance] = None,
    ) -> Scenario:
        """Return a copy carrying a new hazard state.

        Static layers are shared by reference — they are large and immutable in
        practice. This is what makes replay stepping cheap: 48 hourly frames do
        not mean 48 copies of the DEM.
        """
        from dataclasses import replace as _replace

        new = Scenario(
            district_id=self.district_id,
            grid=self.grid,
            dem=self.dem,
            hand=self.hand,
            population=self.population,
            landcover=self.landcover,
            flood_mask=flood_mask,
            depth=depth,
            confidence=confidence,
            roads=self.roads,
            shelters=[_replace(s) for s in self.shelters],
            settlements=[_replace(s) for s in self.settlements],
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            provenance=list(self.provenance),
            meta=dict(self.meta),
        )
        if provenance is not None:
            new.provenance.append(provenance)
        return new

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist to a directory: rasters in one compressed .npz, rest as JSON.

        Compressed npz because these grids are mostly flat floodplain and
        compress ~10x, which matters when the demo laptop is caching several
        replay frames.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        rasters: dict[str, np.ndarray] = {"dem": self.dem, "hand": self.hand, "population": self.population}
        for name in ("landcover", "flood_mask", "depth", "confidence"):
            arr = getattr(self, name)
            if arr is not None:
                rasters[name] = arr
        np.savez_compressed(path / "rasters.npz", **rasters)

        doc = {
            "district_id": self.district_id,
            "grid": self.grid.to_dict(),
            "timestamp": self.timestamp,
            "shelters": [s.to_dict() for s in self.shelters],
            "settlements": [s.to_dict() for s in self.settlements],
            "provenance": [p.to_dict() for p in self.provenance],
            "meta": self.meta,
            "raster_layers": sorted(rasters),
        }
        (path / "scenario.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

        if self.roads is not None:
            _save_graph(self.roads, path / "roads.json")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        path = Path(path)
        doc = json.loads((path / "scenario.json").read_text(encoding="utf-8"))
        with np.load(path / "rasters.npz") as npz:
            rasters = {k: npz[k] for k in npz.files}

        roads = None
        roads_path = path / "roads.json"
        if roads_path.exists():
            roads = _load_graph(roads_path)

        return cls(
            district_id=doc["district_id"],
            grid=Grid.from_dict(doc["grid"]),
            dem=rasters["dem"],
            hand=rasters["hand"],
            population=rasters["population"],
            landcover=rasters.get("landcover"),
            flood_mask=rasters.get("flood_mask"),
            depth=rasters.get("depth"),
            confidence=rasters.get("confidence"),
            roads=roads,
            shelters=[Shelter.from_dict(d) for d in doc.get("shelters", [])],
            settlements=[Settlement.from_dict(d) for d in doc.get("settlements", [])],
            timestamp=doc.get("timestamp", ""),
            provenance=[Provenance(**p) for p in doc.get("provenance", [])],
            meta=doc.get("meta", {}),
        )


# --------------------------------------------------------------------------
# Graph serialisation
# --------------------------------------------------------------------------
# networkx's own JSON helpers churn between versions and choke on numpy
# scalars from osmnx. An explicit format is boring, stable, and debuggable by
# opening the file — which is worth a lot at 2 a.m. before a demo.
# --------------------------------------------------------------------------
def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(i) for k, i in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _save_graph(graph: Any, path: Path) -> None:
    nodes = [{"id": _jsonable(n), **{k: _jsonable(v) for k, v in d.items()}} for n, d in graph.nodes(data=True)]
    edges = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        edges.append(
            {"u": _jsonable(u), "v": _jsonable(v), "key": _jsonable(k), **{a: _jsonable(b) for a, b in d.items()}}
        )
    doc = {"directed": graph.is_directed(), "multigraph": True, "graph": _jsonable(dict(graph.graph)),
           "nodes": nodes, "edges": edges}
    path.write_text(json.dumps(doc), encoding="utf-8")


def _load_graph(path: Path) -> Any:
    import networkx as nx

    doc = json.loads(path.read_text(encoding="utf-8"))
    g = nx.MultiDiGraph() if doc.get("directed", True) else nx.MultiGraph()
    g.graph.update(doc.get("graph", {}))
    for nd in doc["nodes"]:
        nd = dict(nd)
        nid = nd.pop("id")
        g.add_node(nid, **nd)
    for ed in doc["edges"]:
        ed = dict(ed)
        u, v, k = ed.pop("u"), ed.pop("v"), ed.pop("key", 0)
        g.add_edge(u, v, key=k, **ed)
    return g
