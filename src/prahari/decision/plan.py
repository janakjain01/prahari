"""The orchestrator — one flood state in, one evacuation plan out.

    Scenario (with hazard)  ->  build_evacuation_plan()  ->  EvacuationPlan

Order of operations, and why it is this order:

    1. assess shelters      a flooded shelter must be removed BEFORE we
                            compute travel times to it, or we spend the whole
                            optimisation routing people to a swimming pool
    2. annotate roads       depth -> passability -> per-edge travel cost
    3. prune                delete impassable edges
    4. bind                 attach settlements and shelters to the network
    5. travel-time matrix   one Dijkstra per shelter, on the pruned graph
    6. assign               min-cost flow under capacity constraints
    7. route                shortest paths + the counterfactual comparison
    8. prioritise           whoever the plan cannot serve becomes the boat list

Getting 1 before 5 wrong is the single easiest way to produce a plan that
looks complete and is dangerous.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from ..config import DistrictConfig
from ..impact.roads import RoadImpactSummary, annotate_road_passability
from ..impact.population import ExposureResult, compute_evacuation_demand, compute_exposure
from ..impact.shelters import ShelterAssessment, assess_shelters
from ..scenario import Scenario
from .assignment import AssignmentResult, assign_shelters, compute_travel_times
from .graph import EvacuationGraph, TravelMode, bind_features_to_graph, build_evacuation_graph
from .priority import PriorityEntry, rank_priorities
from .routing import Route, compute_routes

logger = logging.getLogger(__name__)

__all__ = ["EvacuationPlan", "build_evacuation_plan"]


@dataclass
class EvacuationPlan:
    """Everything the dashboard, the action cards and the CAP feed need."""

    district_id: str
    district_name: str
    timestamp: str
    mode: str

    exposure: ExposureResult
    roads: RoadImpactSummary
    shelters: ShelterAssessment
    graph: EvacuationGraph
    assignment: AssignmentResult
    routes: list[Route]
    priorities: list[PriorityEntry]

    hazard_method: str = "unknown"
    hazard_detail: str = ""
    is_synthetic: bool = False
    compute_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    evacuation_demand: int = 0
    imminent_band_m: float = 0.0
    # District outline as a closed (lon, lat) ring — used for the CAP <polygon>
    # and for fitting the map. Carried on the plan so the delivery layer never
    # needs the raster grid.
    aoi_polygon: list[tuple[float, float]] = field(default_factory=list)

    # -- headline numbers --------------------------------------------------
    @property
    def people_cut_off(self) -> int:
        from .priority import PriorityReason

        return sum(
            e.people_at_risk for e in self.priorities
            if e.primary_reason is PriorityReason.CUT_OFF
        )

    @property
    def settlements_cut_off(self) -> int:
        from .priority import PriorityReason

        return sum(1 for e in self.priorities if e.primary_reason is PriorityReason.CUT_OFF)

    @property
    def routes_steered(self) -> int:
        """Routes where a flood-blind router would have sent people into water."""
        return sum(1 for r in self.routes if r.naive_route_would_flood)

    def boat_dispatch_list(self, limit: Optional[int] = None) -> list[PriorityEntry]:
        """Settlements with NO road route out, ranked. The boat list.

        Deliberately separated from the capacity-shortfall list. They are two
        different problems handed to two different people: a stranded village
        is an NDRF boat tasking tonight, a capacity shortfall is a request to
        the state for more camps. Merging them into one ranked list — which is
        what a single score naturally does, because a large shortfall outweighs
        a small stranding — buries the taskings that cannot wait.
        """
        from .priority import PriorityReason

        entries = [e for e in self.priorities if e.primary_reason is PriorityReason.CUT_OFF]
        return entries[:limit] if limit else entries

    def capacity_shortfall_list(self, limit: Optional[int] = None) -> list[PriorityEntry]:
        """Settlements that can reach a shelter but find no room in one."""
        from .priority import PriorityReason

        entries = [e for e in self.priorities if e.primary_reason is PriorityReason.UNSHELTERED]
        return entries[:limit] if limit else entries

    def headline(self) -> dict:
        return {
            "exposed_population": round(self.exposure.exposed_population),
            "needing_evacuation": self.evacuation_demand,
            "flooded_area_km2": round(self.exposure.flooded_area_km2, 1),
            "roads_cut_km": round(self.roads.vehicle_blocked_km + self.roads.life_threatening_km, 1),
            "roads_cut_fraction": round(self.roads.cut_fraction, 3),
            "shelters_viable": self.shelters.viable,
            "shelters_flooded_out": self.shelters.flooded_out,
            "people_sheltered": self.assignment.total_sheltered,
            "people_unsheltered": self.assignment.total_unsheltered,
            "settlements_cut_off": self.settlements_cut_off,
            "people_cut_off": self.people_cut_off,
            "mean_journey_minutes": round(self.assignment.mean_travel_minutes, 1),
            "routes_steered_from_flood": self.routes_steered,
        }

    def to_dict(self, include_routes: bool = True) -> dict:
        doc = {
            "district_id": self.district_id,
            "district_name": self.district_name,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "synthetic": self.is_synthetic,
            "compute_seconds": round(self.compute_seconds, 2),
            "hazard": {"method": self.hazard_method, "detail": self.hazard_detail},
            "demand_model": {
                "people_needing_evacuation": self.evacuation_demand,
                "imminent_band_m": self.imminent_band_m,
                "basis": (
                    "People inside the observed flood footprint, attributed to their "
                    "nearest settlement."
                    + (
                        f" Plus people within {self.imminent_band_m:.1f} m of the current "
                        "water surface (anticipatory mode)."
                        if self.imminent_band_m > 0
                        else " Strictly observational — does not anticipate further rise."
                    )
                ),
            },
            "headline": self.headline(),
            "exposure": self.exposure.to_dict(),
            "roads": self.roads.to_dict(),
            "shelter_assessment": self.shelters.to_dict(),
            "graph": self.graph.to_dict(),
            "assignment": self.assignment.to_dict(),
            "priorities": [p.to_dict() for p in self.priorities],
            "boat_dispatch": [p.to_dict() for p in self.boat_dispatch_list()],
            "capacity_shortfall": [p.to_dict() for p in self.capacity_shortfall_list(limit=25)],
            "warnings": self.warnings,
        }
        if include_routes:
            doc["routes"] = [r.to_dict() for r in self.routes]
        return doc


def build_evacuation_plan(
    scenario: Scenario,
    config: DistrictConfig,
    mode: TravelMode = TravelMode.VEHICLE,
    max_routes: Optional[int] = 250,
    compute_counterfactual: bool = True,
    exposure_cell_m: float = 1000.0,
    imminent_band_m: float = 0.0,
) -> EvacuationPlan:
    """Run the full decision pipeline for one flood state.

    ``imminent_band_m``
        Include people who are dry but within this height of the current water
        surface in the evacuation demand. 0 keeps the plan strictly
        observational (move who is already wet); 0.5-1.0 m makes it
        anticipatory. Whichever is used must be stated on the dashboard —
        the two produce very different numbers and a judge will ask which.
    """
    if not scenario.has_hazard:
        raise ValueError(
            "Scenario has no flood mask or depth. Run the hazard layer first "
            "(prahari.hazard.detect_flood)."
        )
    if scenario.roads is None:
        raise ValueError("Scenario has no road network. Run ingestion first.")

    t0 = time.time()
    warnings: list[str] = []
    grid = scenario.grid

    # 1. Shelter viability FIRST — see module docstring.
    shelter_assessment = assess_shelters(
        scenario.shelters, grid, scenario.depth, scenario.flood_mask,
        config.evacuation, config.passability,
    )
    if shelter_assessment.viable == 0:
        warnings.append(
            "EVERY shelter in this district is under water. No road-based evacuation plan "
            "is possible; the entire population becomes an air/boat dispatch problem."
        )

    # 2-3. Roads: annotate, then prune.
    road_summary = annotate_road_passability(
        scenario.roads, grid, scenario.depth, config.passability, config.evacuation
    )
    evac_graph = build_evacuation_graph(scenario.roads, mode=mode)
    if evac_graph.n_components > 1:
        warnings.append(
            f"The road network has fragmented into {evac_graph.n_components} disconnected "
            "components. Settlements outside the component holding a viable shelter cannot "
            "be reached by road."
        )

    # 4. Bind point features to the network.
    bind_features_to_graph(scenario.settlements, evac_graph.graph)
    bind_features_to_graph(scenario.shelters, evac_graph.graph)

    # 5. Travel-time matrix on the pruned graph.
    cutoff = config.evacuation.max_travel_minutes * 60.0 * 3.0
    travel_times = compute_travel_times(
        evac_graph.graph, scenario.settlements, scenario.shelters, mode, cutoff_seconds=cutoff
    )

    # 6. The optimisation — over people who must MOVE, not over everyone.
    total_demand = compute_evacuation_demand(
        grid, scenario.population, scenario.flood_mask, scenario.settlements,
        depth=scenario.depth, hand=scenario.hand,
        include_imminent_band_m=imminent_band_m,
    )
    demand = {s.id: int(round(s.people_at_risk)) for s in scenario.settlements}

    assignment = assign_shelters(
        scenario.settlements, scenario.shelters, travel_times,
        usable_capacity_factor=config.usable_capacity_factor,
        max_travel_seconds=config.evacuation.max_travel_minutes * 60.0,
        demand=demand,
    )
    if assignment.total_unsheltered > 0:
        warnings.append(
            f"{assignment.total_unsheltered:,} people cannot be sheltered with current "
            "capacity. This is a capacity shortfall, not a routing failure."
        )

    # 7. Routes, with the flood-blind counterfactual.
    routes = compute_routes(
        assignment.assignments, evac_graph.graph, scenario.roads,
        scenario.settlements, scenario.shelters, grid, mode,
        compute_counterfactual=compute_counterfactual, max_routes=max_routes,
    )

    # 8. Whoever the plan could not serve.
    priorities = rank_priorities(
        scenario.settlements, scenario.shelters, assignment, travel_times, grid,
        depth=scenario.depth,
        max_travel_seconds=config.evacuation.max_travel_minutes * 60.0,
        life_threatening_depth_m=config.passability.life_threatening_m,
    )

    exposure = compute_exposure(
        grid, scenario.population, scenario.flood_mask, scenario.depth, cell_size_m=exposure_cell_m
    )

    hazard_prov = scenario.provenance_for("hazard")
    plan = EvacuationPlan(
        district_id=scenario.district_id,
        district_name=config.name,
        timestamp=scenario.timestamp or datetime.now(timezone.utc).isoformat(),
        mode=mode.value,
        exposure=exposure,
        roads=road_summary,
        shelters=shelter_assessment,
        graph=evac_graph,
        assignment=assignment,
        routes=routes,
        priorities=priorities,
        hazard_method=hazard_prov.source if hazard_prov else "unknown",
        hazard_detail=hazard_prov.detail if hazard_prov else "",
        is_synthetic=scenario.is_synthetic,
        compute_seconds=time.time() - t0,
        warnings=warnings,
        evacuation_demand=int(round(total_demand)),
        imminent_band_m=imminent_band_m,
        aoi_polygon=grid.polygon_geographic(),
    )

    logger.info(
        "Plan built in %.2fs — %s",
        plan.compute_seconds,
        ", ".join(f"{k}={v}" for k, v in plan.headline().items()),
    )
    return plan
