"""Impact layer — flood x roads, flood x people, flood x shelters.

Three spatial joins. No machine learning, and we do not pretend otherwise.
Bolting a model onto a problem a lookup table solves is something a panel with
a real ML person on it will punish, not reward.

What this layer produces is the input the decision layer needs:

    roads.py       per-edge water depth and a passability class
    population.py  exposed population, aggregated to reporting cells
    shelters.py    which shelters are themselves under water
"""

from .roads import (
    PassabilityStatus,
    annotate_road_passability,
    RoadImpactSummary,
)
from .population import (
    compute_exposure,
    ExposureResult,
    derive_settlements,
)
from .shelters import (
    assess_shelters,
    ShelterAssessment,
    estimate_capacity_from_footprint,
)

__all__ = [
    "PassabilityStatus",
    "annotate_road_passability",
    "RoadImpactSummary",
    "compute_exposure",
    "ExposureResult",
    "derive_settlements",
    "assess_shelters",
    "ShelterAssessment",
    "estimate_capacity_from_footprint",
]
