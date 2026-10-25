"""Priority ranking — the boat dispatch list.

This is the output no existing system produces, and the one a district officer
would actually use at 3 a.m.: an ordered list of settlements that cannot walk
or drive out, worst first.

Everything upstream tells you where the water is. This tells you where to send
the boat.

Reasons a settlement lands on the list, in descending severity:

    CUT_OFF        no road path to ANY viable shelter. Air or boat only.
    UNSHELTERED    a route exists, but every reachable shelter is full.
    DEEP_WATER     already standing in life-threatening depth.
    LONG_JOURNEY   reachable, but beyond the acceptable travel window.
    UNBOUND        not connected to the mapped road network at all — a data
                   gap, flagged honestly rather than silently dropped.

Scoring
-------
    score = people_at_risk x severity_weight x urgency_multiplier

People-weighted on purpose: a cut-off hamlet of 40 and a cut-off town of 4,000
are not the same dispatch decision, and a list sorted purely by severity class
would put them side by side. Depth enters as an urgency multiplier rather than
as its own term, because depth without people is not a rescue priority.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

from ..grid import Grid
from ..scenario import Settlement, Shelter
from .assignment import AssignmentResult

logger = logging.getLogger(__name__)

__all__ = ["PriorityReason", "PriorityEntry", "rank_priorities"]


class PriorityReason(str, Enum):
    CUT_OFF = "cut_off"
    UNSHELTERED = "unsheltered"
    DEEP_WATER = "deep_water"
    LONG_JOURNEY = "long_journey"
    UNBOUND = "unbound_from_network"

    @property
    def weight(self) -> float:
        return {
            PriorityReason.CUT_OFF: 10.0,
            PriorityReason.UNSHELTERED: 7.0,
            PriorityReason.UNBOUND: 6.0,
            PriorityReason.DEEP_WATER: 5.0,
            PriorityReason.LONG_JOURNEY: 2.0,
        }[self]

    @property
    def action(self) -> str:
        return {
            PriorityReason.CUT_OFF: "Dispatch boat — no road route exists",
            PriorityReason.UNSHELTERED: "Open additional capacity or relocate onward",
            PriorityReason.UNBOUND: "Verify access on the ground — settlement is off the mapped network",
            PriorityReason.DEEP_WATER: "Immediate rescue — life-threatening depth at the settlement",
            PriorityReason.LONG_JOURNEY: "Provide transport — journey exceeds the safe window",
        }[self]


@dataclass
class PriorityEntry:
    rank: int
    settlement_id: str
    settlement_name: str
    lon: float
    lat: float
    people_at_risk: int
    population: int
    reasons: list[PriorityReason]
    primary_reason: PriorityReason
    score: float
    depth_m: float
    nearest_shelter_id: Optional[str] = None
    nearest_shelter_minutes: Optional[float] = None
    action: str = ""

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "settlement_id": self.settlement_id,
            "settlement_name": self.settlement_name,
            "lon": round(self.lon, 6),
            "lat": round(self.lat, 6),
            "people_at_risk": self.people_at_risk,
            "population": self.population,
            "reasons": [r.value for r in self.reasons],
            "primary_reason": self.primary_reason.value,
            "score": round(self.score, 1),
            "depth_m": round(self.depth_m, 2),
            "nearest_shelter_id": self.nearest_shelter_id,
            "nearest_shelter_minutes": (
                round(self.nearest_shelter_minutes, 1)
                if self.nearest_shelter_minutes is not None else None
            ),
            "action": self.action,
        }


def rank_priorities(
    settlements: Sequence[Settlement],
    shelters: Sequence[Shelter],
    assignment: AssignmentResult,
    travel_times: dict[tuple[str, str], float],
    grid: Grid,
    depth: Optional[np.ndarray] = None,
    max_travel_seconds: float = 5400.0,
    life_threatening_depth_m: float = 0.60,
    top_n: Optional[int] = None,
) -> list[PriorityEntry]:
    """Build the ranked dispatch list."""
    shelter_map = {s.id: s for s in shelters}
    assigned_people: dict[str, int] = {}
    assigned_minutes: dict[str, float] = {}
    for a in assignment.assignments:
        assigned_people[a.settlement_id] = assigned_people.get(a.settlement_id, 0) + a.people
        assigned_minutes[a.settlement_id] = min(
            assigned_minutes.get(a.settlement_id, math.inf), a.travel_minutes
        )

    reachable: dict[str, tuple[str, float]] = {}
    for (sid, hid), seconds in travel_times.items():
        current = reachable.get(sid)
        if current is None or seconds < current[1]:
            reachable[sid] = (hid, seconds)

    entries: list[PriorityEntry] = []

    for settlement in settlements:
        population = int(round(settlement.population))
        # Ranking is driven by people who need to MOVE, not by how many live
        # there. A dry village with a cut road is a road problem, not a rescue
        # priority, and putting it on the dispatch list buries the villages
        # that are actually in the water.
        at_risk = int(round(settlement.people_at_risk)) or 0
        if population <= 0 or at_risk <= 0:
            continue

        local_depth = _local_depth(settlement, grid, depth)
        unsheltered = int(assignment.unsheltered.get(settlement.id, 0))
        nearest = reachable.get(settlement.id)

        reasons: list[PriorityReason] = []
        people_at_risk = 0

        if settlement.node is None:
            reasons.append(PriorityReason.UNBOUND)
            people_at_risk = at_risk
        elif nearest is None:
            # No viable shelter reachable by any road. This is the headline
            # case: the settlement is an island.
            reasons.append(PriorityReason.CUT_OFF)
            people_at_risk = at_risk

        if unsheltered > 0 and PriorityReason.CUT_OFF not in reasons:
            reasons.append(PriorityReason.UNSHELTERED)
            people_at_risk = max(people_at_risk, unsheltered)

        if local_depth >= life_threatening_depth_m:
            reasons.append(PriorityReason.DEEP_WATER)
            people_at_risk = max(people_at_risk, at_risk)

        if nearest is not None and nearest[1] > max_travel_seconds:
            reasons.append(PriorityReason.LONG_JOURNEY)
            people_at_risk = max(people_at_risk, assigned_people.get(settlement.id, at_risk))

        if not reasons:
            continue

        primary = max(reasons, key=lambda r: r.weight)

        # Urgency rises with depth but saturates: past knee height the
        # difference between 0.9 m and 1.4 m does not change the dispatch
        # decision, and letting it scale without limit would let one deep
        # hamlet outrank a large cut-off town.
        urgency = 1.0 + min(local_depth / life_threatening_depth_m, 2.0)
        score = people_at_risk * primary.weight * urgency

        lon, lat = grid.to_geographic(settlement.x, settlement.y)
        entries.append(
            PriorityEntry(
                rank=0,
                settlement_id=settlement.id,
                settlement_name=settlement.name,
                lon=float(lon), lat=float(lat),
                people_at_risk=int(people_at_risk),
                population=population,
                reasons=reasons,
                primary_reason=primary,
                score=score,
                depth_m=local_depth,
                nearest_shelter_id=nearest[0] if nearest else None,
                nearest_shelter_minutes=nearest[1] / 60.0 if nearest else None,
                action=primary.action,
            )
        )

    entries.sort(key=lambda e: (-e.score, -e.people_at_risk))
    for i, entry in enumerate(entries, start=1):
        entry.rank = i

    if top_n:
        entries = entries[:top_n]

    cut_off = [e for e in entries if e.primary_reason is PriorityReason.CUT_OFF]
    logger.info(
        "Priority list: %d settlements flagged, %d completely CUT OFF (%d people needing "
        "boat or air rescue)",
        len(entries), len(cut_off), sum(e.people_at_risk for e in cut_off),
    )
    return entries


def _local_depth(settlement: Settlement, grid: Grid, depth: Optional[np.ndarray]) -> float:
    """Depth at the settlement, sampled over a small neighbourhood."""
    if depth is None:
        return float(settlement.flood_depth_m or 0.0)
    row, col = grid.rowcol(settlement.x, settlement.y)
    r, c = int(row), int(col)
    if not (0 <= r < grid.height and 0 <= c < grid.width):
        return 0.0
    r0, r1 = max(r - 2, 0), min(r + 3, grid.height)
    c0, c1 = max(c - 2, 0), min(c + 3, grid.width)
    patch = np.nan_to_num(depth[r0:r1, c0:c1])
    # 75th percentile: responsive to real inundation, robust to one hot pixel.
    return float(np.percentile(patch, 75)) if patch.size else 0.0
