"""Citizen action card — one screen, no login, readable in three seconds.

The documented failure of existing alerting is not delivery, it is ALERT
FATIGUE. SACHET has pushed billions of messages. People stop reading them
because a district-level "heavy rainfall likely" message tells them nothing
they can act on.

So the card obeys four rules, and every one of them is a constraint we can be
held to:

  1. ONE destination. Not a list of options. A person deciding under stress
     with a rising river does not want to compare shelters.
  2. ONE route instruction, phrased as a road name, not a bearing or a
     distance-to-turn. "Via Jania, not the main road" is how people actually
     give directions here.
  3. The NEGATIVE instruction is explicit. Telling someone where to go without
     telling them which road is a river leaves them to discover it.
  4. It fits on one screen of a cheap phone, in the reader's language.

The stranded case gets a DIFFERENT card, not a degraded version of the same
one. Telling someone to walk 1.4 km when every road is submerged is worse than
telling them nothing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from ..decision.plan import EvacuationPlan
from ..decision.priority import PriorityEntry, PriorityReason
from ..decision.routing import Route
from .i18n import LANGUAGES, translate

logger = logging.getLogger(__name__)

__all__ = ["ActionCard", "build_action_card", "build_stranded_card", "build_all_cards"]


@dataclass
class ActionCard:
    """One citizen-facing message, in one language."""

    settlement_id: str
    settlement_name: str
    language: str
    kind: str                       # "evacuate" | "stranded"
    title: str
    destination: str
    distance_line: str
    route_line: str
    warning_line: str
    carry_line: str
    livestock_line: str
    banner: str
    issued: str
    valid_until: str
    people: int = 0
    lon: Optional[float] = None
    lat: Optional[float] = None
    shelter_lon: Optional[float] = None
    shelter_lat: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Plain-text rendering — this is what goes down an SMS or cell broadcast.

        Kept deliberately narrow. Cell broadcast and feature-phone SMS are how
        this reaches the people least likely to have a smartphone, which in
        rural Assam is most of them.
        """
        lines = [f"[{self.banner}]", "", self.title, ""]
        if self.kind == "evacuate":
            lines += [f"{translate('go_to', self.language)}: {self.destination}", self.distance_line]
            if self.route_line:
                lines.append(self.route_line)
        else:
            lines.append(self.destination)
        if self.warning_line:
            lines += ["", self.warning_line]
        lines += ["", self.carry_line, self.livestock_line]
        return "\n".join(line for line in lines if line is not None)


def _banner(plan: EvacuationPlan, language: str) -> str:
    if plan.is_synthetic:
        return translate("synthetic_banner", language)
    return translate("exercise_banner", language)


def _validity(plan: EvacuationPlan, hours: int = 6) -> tuple[str, str]:
    try:
        issued = datetime.fromisoformat(plan.timestamp)
    except ValueError:
        issued = datetime.now(timezone.utc)
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)
    return issued.isoformat(), (issued + timedelta(hours=hours)).isoformat()


def build_action_card(
    route: Route,
    plan: EvacuationPlan,
    language: str = "en",
    shelter_utilisation: Optional[float] = None,
    shelter_coords: Optional[tuple[float, float]] = None,
    settlement_coords: Optional[tuple[float, float]] = None,
) -> ActionCard:
    """Build the evacuation card for one route."""
    issued, valid_until = _validity(plan)

    if route.mode == "foot":
        distance_line = translate(
            "distance_walk", language,
            km=f"{route.distance_km:.1f}", minutes=int(round(route.travel_minutes)),
        )
    else:
        distance_line = translate(
            "distance_vehicle", language,
            km=f"{route.distance_km:.1f}", minutes=int(round(route.travel_minutes)),
        )

    # Route instruction. The "avoid" half only appears when we genuinely
    # steered around something — inventing a warning erodes trust in every
    # future message.
    if route.avoided_roads:
        via = route.road_names[0] if route.road_names else ""
        avoid = route.avoided_roads[0]
        if via and via != avoid:
            route_line = (
                translate("route_via", language, via=via)
                + " — "
                + translate("route_avoid", language, avoid=avoid)
            )
        else:
            route_line = translate("route_avoid", language, avoid=avoid)
    elif route.road_names:
        route_line = translate("route_via", language, via=route.road_names[0])
    else:
        route_line = translate("route_plain", language)

    warnings: list[str] = []
    if route.max_depth_on_route_m >= 0.15:
        warnings.append(
            translate("depth_warning", language, depth=f"{route.max_depth_on_route_m:.1f}")
        )
    if shelter_utilisation is not None and shelter_utilisation >= 0.85:
        warnings.append(translate("shelter_full_warning", language))

    return ActionCard(
        settlement_id=route.settlement_id,
        settlement_name=route.settlement_name,
        language=language,
        kind="evacuate",
        title=translate("alert_title", language),
        destination=route.shelter_name,
        distance_line=distance_line,
        route_line=route_line,
        warning_line=" ".join(warnings),
        carry_line=translate("carry", language),
        livestock_line=translate("livestock", language),
        banner=_banner(plan, language),
        issued=issued,
        valid_until=valid_until,
        people=route.people,
        lon=settlement_coords[0] if settlement_coords else None,
        lat=settlement_coords[1] if settlement_coords else None,
        shelter_lon=shelter_coords[0] if shelter_coords else None,
        shelter_lat=shelter_coords[1] if shelter_coords else None,
    )


def build_stranded_card(
    entry: PriorityEntry,
    plan: EvacuationPlan,
    language: str = "en",
) -> ActionCard:
    """Card for a settlement with no road route out.

    A different message, not a softened one. "Do not try to leave by road" is
    the instruction that keeps people out of the water while the boat is
    tasked.
    """
    issued, valid_until = _validity(plan)
    warning = ""
    if entry.depth_m >= 0.15:
        warning = translate("depth_warning", language, depth=f"{entry.depth_m:.1f}")

    return ActionCard(
        settlement_id=entry.settlement_id,
        settlement_name=entry.settlement_name,
        language=language,
        kind="stranded",
        title=translate("no_route_title", language),
        destination=translate("no_route_body", language),
        distance_line="",
        route_line="",
        warning_line=warning,
        carry_line=translate("carry", language),
        livestock_line=translate("livestock", language),
        banner=_banner(plan, language),
        issued=issued,
        valid_until=valid_until,
        people=entry.people_at_risk,
        lon=entry.lon,
        lat=entry.lat,
    )


def build_all_cards(
    plan: EvacuationPlan,
    languages: Sequence[str] = LANGUAGES,
    limit: Optional[int] = None,
) -> list[ActionCard]:
    """Build every card the plan implies, in every requested language.

    Stranded settlements come FIRST. If anything downstream truncates the list
    — an SMS gateway rate limit, a paginated UI — the people who cannot move
    must not be the ones cut off.
    """
    cards: list[ActionCard] = []
    utilisation = {sid: plan.assignment.utilisation(sid) for sid in plan.assignment.shelter_load}

    for entry in plan.boat_dispatch_list(limit=limit):
        for language in languages:
            cards.append(build_stranded_card(entry, plan, language))

    routes = plan.routes[:limit] if limit else plan.routes
    for route in routes:
        shelter_coords = route.coordinates[-1] if route.coordinates else None
        settlement_coords = route.coordinates[0] if route.coordinates else None
        for language in languages:
            cards.append(
                build_action_card(
                    route, plan, language,
                    shelter_utilisation=utilisation.get(route.shelter_id),
                    shelter_coords=shelter_coords,
                    settlement_coords=settlement_coords,
                )
            )

    logger.info(
        "Built %d action cards (%d settlements x %d languages)",
        len(cards), len(cards) // max(len(languages), 1), len(languages),
    )
    return cards
