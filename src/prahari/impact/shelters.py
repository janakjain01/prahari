"""Shelter viability — a shelter standing in floodwater is not a shelter.

This check sounds trivially obvious, which is exactly why it is worth having:
it is the detail that makes a demo feel like it was built by people who
thought about the actual event rather than about the slide. Assam relief camps
are overwhelmingly schools, schools are sited in villages, and villages are
sited on the floodplain. Some of them go under.

Capacity honesty
----------------
Surveyed shelter capacity is not open data. We therefore report capacity with
its PROVENANCE attached, and the dashboard renders estimated figures
differently from surveyed ones:

    surveyed            from an official relief-camp list, if we have one
    footprint_estimate  building footprint area / m2 per person
    default_assumption  a flat fallback where we have neither

Density: the Sphere Handbook minimum standard is 3.5 m2 of covered floor area
per person. Indian relief camps under surge routinely run tighter than that.
We use 3.5 and label it, rather than quietly using a number that makes our
capacity look better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import EvacuationConfig, PassabilityConfig
from ..grid import Grid
from ..scenario import Shelter

logger = logging.getLogger(__name__)

__all__ = ["assess_shelters", "ShelterAssessment", "estimate_capacity_from_footprint"]


@dataclass
class ShelterAssessment:
    total: int
    viable: int
    flooded_out: int
    total_capacity: int
    viable_capacity: int
    usable_capacity: int
    estimated_capacity_fraction: float

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "viable": self.viable,
            "flooded_out": self.flooded_out,
            "total_capacity": self.total_capacity,
            "viable_capacity": self.viable_capacity,
            "usable_capacity": self.usable_capacity,
            "estimated_capacity_fraction": round(self.estimated_capacity_fraction, 3),
            "capacity_caveat": (
                "Capacity for some shelters is ESTIMATED from building footprint at "
                "3.5 m2/person (Sphere minimum), not surveyed. Production deployment "
                "needs the district relief-camp register."
            ),
        }


def estimate_capacity_from_footprint(
    footprint_sqm: float,
    sqm_per_person: float = 3.5,
    usable_floor_fraction: float = 0.65,
    storeys: float = 1.0,
) -> int:
    """Convert a building footprint into a defensible occupancy figure.

    ``usable_floor_fraction`` accounts for the fact that a school's footprint
    is not one open hall: corridors, walls, toilets and staff rooms are not
    sleeping space. 0.65 is deliberately conservative — over-estimating
    shelter capacity is the error that leaves people standing outside a full
    building in the rain.
    """
    if footprint_sqm <= 0:
        return 0
    usable = footprint_sqm * usable_floor_fraction * max(storeys, 1.0)
    return max(int(usable / sqm_per_person), 0)


def assess_shelters(
    shelters: Sequence[Shelter],
    grid: Grid,
    depth: np.ndarray,
    flood_mask: np.ndarray,
    evacuation: EvacuationConfig,
    passability: PassabilityConfig,
    plinth_height_m: float = 0.30,
    sample_radius_px: int = 2,
) -> ShelterAssessment:
    """Mark each shelter viable or flooded out. Mutates the shelters in place.

    ``plinth_height_m``
        Public buildings in flood-prone Assam are routinely built on a raised
        plinth, so ground-level water does not immediately mean water inside
        the building. We credit 0.30 m of plinth before declaring a shelter
        inundated. This is a genuine local-practice adjustment, and also the
        kind of assumption that must be stated rather than hidden — it is in
        the API response.

    ``sample_radius_px``
        Depth is sampled over a small neighbourhood, not a single pixel. A
        shelter's usefulness depends on its surroundings being reachable, and
        a single-pixel sample is hostage to one noisy cell.
    """
    dep = np.nan_to_num(np.asarray(depth, dtype=np.float64))
    mask = np.asarray(flood_mask) > 0

    viable = flooded_out = 0
    total_capacity = viable_capacity = 0
    estimated = 0

    for shelter in shelters:
        row, col = grid.rowcol(shelter.x, shelter.y)
        r, c = int(row), int(col)
        if not (0 <= r < grid.height and 0 <= c < grid.width):
            # Outside the AOI: we cannot assess it, so we must not offer it.
            shelter.flood_depth_m = 0.0
            shelter.is_viable = False
            flooded_out += 1
            total_capacity += shelter.capacity
            continue

        r0, r1 = max(r - sample_radius_px, 0), min(r + sample_radius_px + 1, grid.height)
        c0, c1 = max(c - sample_radius_px, 0), min(c + sample_radius_px + 1, grid.width)
        patch_depth = dep[r0:r1, c0:c1]
        patch_mask = mask[r0:r1, c0:c1]

        # Median over the wet part of the neighbourhood: robust to one bad
        # pixel, but still responds as soon as the site genuinely goes under.
        wet = patch_depth[patch_mask]
        local_depth = float(np.median(wet)) if wet.size else 0.0
        shelter.flood_depth_m = round(local_depth, 3)

        effective = local_depth - plinth_height_m
        shelter.is_viable = effective < passability.clear_m

        total_capacity += shelter.capacity
        if shelter.is_viable:
            viable += 1
            viable_capacity += shelter.capacity
        else:
            flooded_out += 1
        if shelter.capacity_source != "surveyed":
            estimated += 1

    usable = int(viable_capacity * (1.0 - evacuation.capacity_reserve_fraction))

    assessment = ShelterAssessment(
        total=len(shelters),
        viable=viable,
        flooded_out=flooded_out,
        total_capacity=total_capacity,
        viable_capacity=viable_capacity,
        usable_capacity=usable,
        estimated_capacity_fraction=estimated / len(shelters) if shelters else 0.0,
    )
    logger.info(
        "Shelters: %d viable of %d (%d flooded out); usable capacity %d after a %.0f%% reserve",
        viable, len(shelters), flooded_out, usable, 100 * evacuation.capacity_reserve_fraction,
    )
    if flooded_out:
        logger.warning(
            "%d shelter(s) are themselves under water and have been removed from the "
            "assignment. Routing people to a flooded shelter is the failure mode this "
            "check exists to prevent.", flooded_out,
        )
    return assessment
