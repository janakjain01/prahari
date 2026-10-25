"""Decision layer — the part nobody else builds.

Everything upstream of this exists somewhere already. IMD and CWC forecast.
C-FLOOD models three basins. Google Flood Hub does riverine forecasting better
than we could. SACHET delivers alerts to 44 billion phones.

None of them answer the question a family actually has: WHICH ROAD.

    graph.py       prune flooded edges out of the road network
    assignment.py  capacity-constrained shelter assignment (min-cost flow)
    routing.py     safe route per settlement, on the pruned graph
    priority.py    rank settlements with NO route — the boat dispatch list
    plan.py        orchestrates the above into one EvacuationPlan

The line that wins the room:

    "Google Maps will route you straight into the water. We route you around
     it — and we make sure the shelter we send you to still has room."

Nearest-shelter routing is the naive answer and it is wrong: it overflows the
nearest shelter while a viable one two kilometres away sits empty. Assignment
under capacity constraints is an optimisation problem, and treating it as one
is the technical core of this project.
"""

from .graph import (
    build_evacuation_graph,
    bind_features_to_graph,
    EvacuationGraph,
    TravelMode,
)
from .assignment import (
    assign_shelters,
    AssignmentResult,
    Assignment,
)
from .routing import compute_routes, Route
from .priority import rank_priorities, PriorityEntry, PriorityReason
from .plan import build_evacuation_plan, EvacuationPlan

__all__ = [
    "build_evacuation_graph",
    "bind_features_to_graph",
    "EvacuationGraph",
    "TravelMode",
    "assign_shelters",
    "AssignmentResult",
    "Assignment",
    "compute_routes",
    "Route",
    "rank_priorities",
    "PriorityEntry",
    "PriorityReason",
    "build_evacuation_plan",
    "EvacuationPlan",
]
