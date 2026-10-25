"""Graph pruning — delete the roads that are under water.

Step 1 of the decision layer, and the cheapest idea in the project to explain:
take the road network, remove every edge the flood has made impassable, and do
all subsequent routing on what is left. A router that does not do this will
happily send a family down a road that is a river.

Two subtleties that a careful reader will ask about:

  * MODE MATTERS. Vehicles are stopped at ~0.30 m; pedestrians keep going to
    ~0.60 m. Those are different graphs, so we build them separately. Most of
    rural Assam evacuates on foot, so the foot graph is not an afterthought.

  * BINDING IS A FAILURE MODE. Settlements and shelters are points; the graph
    has nodes. Snapping a shelter to the nearest node is easy and quietly
    wrong when the nearest node sits on the far side of a flooded channel. We
    snap to the nearest node, record the snap distance, and flag any binding
    beyond a threshold rather than pretending the shelter is on the road.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TravelMode",
    "EvacuationGraph",
    "build_evacuation_graph",
    "bind_features_to_graph",
    "BindingReport",
]


class TravelMode(str, Enum):
    VEHICLE = "vehicle"
    FOOT = "foot"

    @property
    def weight_attribute(self) -> str:
        return "travel_time_s" if self is TravelMode.VEHICLE else "walk_time_s"


@dataclass
class BindingReport:
    """How well point features attached to the road network."""

    bound: int = 0
    unbound: int = 0
    max_snap_m: float = 0.0
    mean_snap_m: float = 0.0
    far_snaps: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bound": self.bound,
            "unbound": self.unbound,
            "max_snap_m": round(self.max_snap_m, 1),
            "mean_snap_m": round(self.mean_snap_m, 1),
            "far_snaps": [{"id": i, "snap_m": round(d, 1)} for i, d in self.far_snaps[:20]],
        }


@dataclass
class EvacuationGraph:
    """A road network with the flooded edges removed."""

    graph: Any                       # networkx.MultiDiGraph — pruned
    mode: TravelMode
    original_edges: int
    retained_edges: int
    removed_edges: int
    removed_km: float
    n_components: int
    largest_component_fraction: float

    @property
    def weight(self) -> str:
        return self.mode.weight_attribute

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "original_edges": self.original_edges,
            "retained_edges": self.retained_edges,
            "removed_edges": self.removed_edges,
            "removed_km": round(self.removed_km, 2),
            "removed_fraction": round(
                self.removed_edges / self.original_edges if self.original_edges else 0.0, 4
            ),
            "n_components": self.n_components,
            "largest_component_fraction": round(self.largest_component_fraction, 4),
        }


def build_evacuation_graph(
    roads: Any,
    mode: TravelMode = TravelMode.VEHICLE,
    keep_isolated_nodes: bool = True,
) -> EvacuationGraph:
    """Return a copy of the road graph with impassable edges deleted.

    ``keep_isolated_nodes`` is True by default and that is deliberate: a node
    stranded with no usable edge is not noise to be tidied away, it is a
    settlement that has been CUT OFF. Deleting it would erase exactly the
    signal the priority list is built from.
    """
    import networkx as nx

    weight_attr = mode.weight_attribute
    pruned = nx.MultiDiGraph()
    pruned.graph.update(getattr(roads, "graph", {}))
    pruned.add_nodes_from(roads.nodes(data=True))

    original = 0
    removed = 0
    removed_m = 0.0

    for u, v, k, data in roads.edges(keys=True, data=True):
        original += 1
        cost = data.get(weight_attr)
        if cost is None:
            # No flood annotation on this edge: fall back to free-flow length.
            # Treating an un-annotated edge as blocked would silently delete
            # the whole network in the no-flood case.
            length = float(data.get("length", 0.0) or 0.0)
            cost = length / 6.94 if mode is TravelMode.VEHICLE else length / 1.11
        if not math.isfinite(float(cost)):
            removed += 1
            removed_m += float(data.get("length", 0.0) or 0.0)
            continue
        pruned.add_edge(u, v, key=k, **{**data, weight_attr: float(cost)})

    if not keep_isolated_nodes:
        pruned.remove_nodes_from([n for n in list(pruned.nodes) if pruned.degree(n) == 0])

    undirected = pruned.to_undirected(as_view=False)
    components = list(nx.connected_components(undirected)) if undirected.number_of_nodes() else []
    largest = max((len(c) for c in components), default=0)
    frac = largest / pruned.number_of_nodes() if pruned.number_of_nodes() else 0.0

    result = EvacuationGraph(
        graph=pruned,
        mode=mode,
        original_edges=original,
        retained_edges=pruned.number_of_edges(),
        removed_edges=removed,
        removed_km=removed_m / 1000.0,
        n_components=len(components),
        largest_component_fraction=frac,
    )
    logger.info(
        "Pruned graph (%s): removed %d/%d edges (%.1f km); network fell into %d components, "
        "largest holds %.1f%% of nodes",
        mode.value, removed, original, result.removed_km, result.n_components, 100 * frac,
    )
    if result.n_components > 1:
        logger.warning(
            "The road network is FRAGMENTED into %d components. Settlements outside the "
            "component containing a viable shelter cannot be reached by road at all — "
            "they are the boat dispatch list.", result.n_components,
        )
    return result


def bind_features_to_graph(
    features: Sequence[Any],
    graph: Any,
    max_snap_m: float = 2000.0,
    warn_snap_m: float = 500.0,
) -> BindingReport:
    """Attach point features to their nearest graph node, in place.

    Sets ``feature.node``. Features farther than ``max_snap_m`` from any node
    are left unbound (``node = None``) rather than snapped across the district
    — an unbound shelter is honest, a wrongly-snapped one is a routing lie.

    Uses a KD-tree; a linear scan over ~50k nodes for each of ~500 features is
    25M distance computations per cycle, which is the sort of thing that turns
    a two-second refresh into a thirty-second one.
    """
    report = BindingReport()
    if not features:
        # A district with no mapped shelters is a real, reportable state — the
        # plan then correctly shows everyone as unshelterable. It must not be
        # a crash inside the KD-tree.
        logger.warning("bind_features_to_graph called with no features — nothing to bind")
        return report

    nodes = [(n, d) for n, d in graph.nodes(data=True) if "x" in d and "y" in d]
    if not nodes:
        logger.error("Road graph has no positioned nodes — cannot bind features")
        for f in features:
            f.node = None
        report.unbound = len(features)
        return report

    node_ids = [n for n, _ in nodes]
    coords = np.array([[float(d["x"]), float(d["y"])] for _, d in nodes])

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        query = np.array([[float(f.x), float(f.y)] for f in features])
        distances, indices = tree.query(query, k=1)
    except ImportError:  # pragma: no cover - scipy is a hard dep in practice
        distances, indices = _brute_force_nearest(features, coords)

    snaps: list[float] = []
    for feature, dist, idx in zip(features, np.atleast_1d(distances), np.atleast_1d(indices)):
        dist = float(dist)
        if dist > max_snap_m:
            feature.node = None
            report.unbound += 1
            logger.warning(
                "Feature %s is %.0f m from the nearest road node (limit %.0f m) and was left "
                "unbound rather than snapped across country.",
                getattr(feature, "id", "?"), dist, max_snap_m,
            )
            continue
        feature.node = node_ids[int(idx)]
        report.bound += 1
        snaps.append(dist)
        if dist > warn_snap_m:
            report.far_snaps.append((str(getattr(feature, "id", "?")), dist))

    if snaps:
        report.max_snap_m = max(snaps)
        report.mean_snap_m = float(np.mean(snaps))
    logger.info(
        "Bound %d/%d features to the road network (mean snap %.0f m, max %.0f m)",
        report.bound, len(features), report.mean_snap_m, report.max_snap_m,
    )
    return report


def _brute_force_nearest(features, coords):  # pragma: no cover - fallback
    dists, idxs = [], []
    for f in features:
        d2 = (coords[:, 0] - f.x) ** 2 + (coords[:, 1] - f.y) ** 2
        i = int(np.argmin(d2))
        dists.append(math.sqrt(float(d2[i])))
        idxs.append(i)
    return np.array(dists), np.array(idxs)
