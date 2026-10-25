"""CAP 1.2 export — how PRAHARI plugs into the national system instead of
competing with it.

CAP (Common Alerting Protocol, OASIS 1.2) is the schema SACHET itself uses.
Emitting valid CAP costs an afternoon and it converts the single biggest
objection this project faces —

    "Isn't this a parallel system to what NDMA already runs?"

— into a credibility slide:

    "No. We consume IMD and CWC forecasts, and we publish our output back in
     SACHET's own CAP 1.2 format. We are a decision layer inside the national
     pipeline, not beside it. It also answers your other question: CAP is what
     drives cell broadcast, so this reaches a feature phone with no app and no
     internet."

Details that matter for real interoperability, all of which are easy to get
wrong and all of which a CAP-aware reviewer will check:

  * ELEMENT ORDER IS PART OF THE SCHEMA. CAP is xsd:sequence, not a bag of
    tags. Emitting <expires> before <effective> is invalid, and validators
    reject it.
  * POLYGON COORDINATES ARE "lat,lon", not "lon,lat" — the opposite of
    GeoJSON. The ring must be explicitly closed.
  * <identifier> must not contain spaces, commas or restricted characters.
  * Timestamps need a numeric UTC offset. A trailing "Z" is not valid CAP;
    India Standard Time is +05:30 and that is what we emit.
  * status="Exercise" for anything that is not an authorised live alert. This
    is enforced in config.py and cannot be flipped by accident.
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from ..config import DistrictConfig
from ..decision.plan import EvacuationPlan
from ..decision.priority import PriorityEntry
from .i18n import LANGUAGE_TAGS, translate

logger = logging.getLogger(__name__)

__all__ = ["build_cap_alert", "cap_to_string", "CAPAlert", "validate_cap"]

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
IST = timezone(timedelta(hours=5, minutes=30))

_IDENTIFIER_SAFE = re.compile(r"[^A-Za-z0-9._:-]")


@dataclass
class CAPAlert:
    identifier: str
    xml: str
    language_count: int
    area_count: int
    status: str

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "language_count": self.language_count,
            "area_count": self.area_count,
            "status": self.status,
            "bytes": len(self.xml.encode("utf-8")),
        }


def _cap_time(dt: datetime) -> str:
    """Format for CAP: ISO 8601 with a NUMERIC offset, never 'Z'.

    ``strftime('%z')`` yields ``+0530``; CAP wants ``+05:30``.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    stamp = dt.astimezone(IST).strftime("%Y-%m-%dT%H:%M:%S%z")
    return f"{stamp[:-2]}:{stamp[-2:]}"


def _sanitise_identifier(raw: str) -> str:
    return _IDENTIFIER_SAFE.sub("-", raw)


def _sub(parent: ET.Element, tag: str, text: Optional[str] = None) -> ET.Element:
    el = ET.SubElement(parent, tag)
    if text is not None:
        el.text = str(text)
    return el


def _geocode(parent: ET.Element, name: str, value: str) -> None:
    gc = _sub(parent, "geocode")
    _sub(gc, "valueName", name)
    _sub(gc, "value", value)


def _parameter(parent: ET.Element, name: str, value: str) -> None:
    p = _sub(parent, "parameter")
    _sub(p, "valueName", name)
    _sub(p, "value", str(value))


def build_cap_alert(
    plan: EvacuationPlan,
    config: DistrictConfig,
    languages: Optional[Sequence[str]] = None,
    expires_hours: int = 6,
    web: str = "",
    msg_type: str = "Alert",
) -> CAPAlert:
    """Render one CAP 1.2 alert covering the whole district.

    One ``<info>`` block per language, as CAP intends — a single alert carrying
    Assamese, Bengali, Hindi and English is one message to SACHET, not four.
    """
    languages = list(languages or config.alerting.languages)
    cap_cfg = config.alerting.cap

    try:
        sent_dt = datetime.fromisoformat(plan.timestamp)
    except ValueError:
        sent_dt = datetime.now(timezone.utc)
    if sent_dt.tzinfo is None:
        sent_dt = sent_dt.replace(tzinfo=timezone.utc)
    expires_dt = sent_dt + timedelta(hours=expires_hours)

    digest = hashlib.sha256(
        f"{config.id}|{plan.timestamp}|{plan.evacuation_demand}".encode("utf-8")
    ).hexdigest()[:12]
    identifier = _sanitise_identifier(f"PRAHARI-{config.id}-{sent_dt.strftime('%Y%m%dT%H%M%S')}-{digest}")

    ET.register_namespace("", CAP_NS)
    alert = ET.Element(f"{{{CAP_NS}}}alert")

    # --- alert-level elements, in schema order ---------------------------
    _sub(alert, "identifier", identifier)
    _sub(alert, "sender", cap_cfg.sender)
    _sub(alert, "sent", _cap_time(sent_dt))
    _sub(alert, "status", cap_cfg.status)
    _sub(alert, "msgType", msg_type)
    _sub(alert, "scope", cap_cfg.scope)
    _sub(
        alert, "note",
        "Generated by PRAHARI, a flood evacuation decision layer. "
        + (
            "SYNTHETIC DEMONSTRATION DATA — not a real flood."
            if plan.is_synthetic
            else "Prototype output — not an authorised operational alert."
        ),
    )

    headline_numbers = plan.headline()

    for language in languages:
        info = _sub(alert, "info")
        _sub(info, "language", LANGUAGE_TAGS.get(language, language))
        _sub(info, "category", "Met")
        _sub(info, "event", "Flood — evacuation routing")
        _sub(info, "responseType", "Evacuate")
        _sub(info, "urgency", cap_cfg.urgency)
        _sub(info, "severity", cap_cfg.severity)
        _sub(info, "certainty", cap_cfg.certainty)

        ec = _sub(info, "eventCode")
        _sub(ec, "valueName", "SAME")
        _sub(ec, "value", "FLW")           # Flood Warning

        _sub(info, "effective", _cap_time(sent_dt))
        _sub(info, "onset", _cap_time(sent_dt))
        _sub(info, "expires", _cap_time(expires_dt))
        _sub(info, "senderName", cap_cfg.sender_name)
        _sub(info, "headline", _headline_text(plan, config, language))
        _sub(info, "description", _description_text(plan, config, language))
        _sub(info, "instruction", _instruction_text(plan, language))
        if web:
            _sub(info, "web", web)

        # Machine-readable payload. This is the part that makes the alert
        # actionable by another system rather than only readable by a person.
        _parameter(info, "PRAHARI_people_needing_evacuation", headline_numbers["needing_evacuation"])
        _parameter(info, "PRAHARI_people_sheltered", headline_numbers["people_sheltered"])
        _parameter(info, "PRAHARI_people_unsheltered", headline_numbers["people_unsheltered"])
        _parameter(info, "PRAHARI_settlements_cut_off", headline_numbers["settlements_cut_off"])
        _parameter(info, "PRAHARI_people_cut_off", headline_numbers["people_cut_off"])
        _parameter(info, "PRAHARI_roads_cut_km", headline_numbers["roads_cut_km"])
        _parameter(info, "PRAHARI_shelters_viable", headline_numbers["shelters_viable"])
        _parameter(info, "PRAHARI_hazard_method", plan.hazard_method)
        _parameter(info, "PRAHARI_data_basis", "SYNTHETIC" if plan.is_synthetic else "OBSERVED")

        # --- areas -------------------------------------------------------
        district_area = _sub(info, "area")
        _sub(district_area, "areaDesc", f"{config.name} district, {config.district.state}")
        _sub(district_area, "polygon", _polygon_from_plan(plan))
        if config.district.lgd_code:
            _geocode(district_area, "LGD", config.district.lgd_code)
        if config.district.census_code:
            _geocode(district_area, "CensusStateCode", config.district.census_code)

        # A separate <area> per cut-off settlement, so a downstream cell
        # broadcast can geofence the "do not travel by road" instruction to
        # exactly the people it applies to instead of alarming the district.
        for entry in plan.boat_dispatch_list(limit=40):
            area = _sub(info, "area")
            _sub(area, "areaDesc", f"{entry.settlement_name} (no road route — boat dispatch)")
            _sub(area, "circle", f"{entry.lat:.6f},{entry.lon:.6f} 1.5")
            _geocode(area, "PRAHARI_settlement_id", entry.settlement_id)

    xml = cap_to_string(alert)
    # Counted with the bare tag: only the ROOT carries the namespace in the
    # ElementTree object (children inherit it via the default xmlns only once
    # serialised), so a namespaced findall here silently returns nothing.
    area_count = len(alert.findall(".//area"))

    logger.info(
        "CAP 1.2 alert %s: %d languages, %d areas, status=%s, %d bytes",
        identifier, len(languages), area_count, cap_cfg.status, len(xml.encode("utf-8")),
    )
    return CAPAlert(
        identifier=identifier, xml=xml, language_count=len(languages),
        area_count=area_count, status=cap_cfg.status,
    )


def _polygon_from_plan(plan: EvacuationPlan) -> str:
    """District outline as a CAP polygon: space-separated ``lat,lon`` pairs.

    Note the axis order. CAP is lat,lon; GeoJSON is lon,lat. Swapping them
    silently relocates the alert to a different hemisphere, which is the kind
    of bug that survives testing because the XML still validates.
    """
    if plan.aoi_polygon:
        return " ".join(f"{lat:.6f},{lon:.6f}" for lon, lat in plan.aoi_polygon)
    # Fallback: envelope of the exposure cells.
    lons = [c.lon for c in plan.exposure.cells] or [0.0]
    lats = [c.lat for c in plan.exposure.cells] or [0.0]
    w, e = min(lons), max(lons)
    s, n = min(lats), max(lats)
    corners = [(w, n), (e, n), (e, s), (w, s), (w, n)]
    return " ".join(f"{lat:.6f},{lon:.6f}" for lon, lat in corners)


def _headline_text(plan: EvacuationPlan, config: DistrictConfig, language: str) -> str:
    h = plan.headline()
    if language == "en":
        return (
            f"Flood evacuation advisory for {config.name}: {h['needing_evacuation']:,} people "
            f"need to move, {h['roads_cut_km']} km of road impassable, "
            f"{h['settlements_cut_off']} settlements with no road route."
        )
    return translate("alert_title", language)


def _description_text(plan: EvacuationPlan, config: DistrictConfig, language: str) -> str:
    h = plan.headline()
    if language != "en":
        return translate("alert_title", language)
    lines = [
        f"{h['flooded_area_km2']} km2 of {config.name} district is inundated.",
        f"{h['exposed_population']:,} people are within the observed flood footprint.",
        f"{h['roads_cut_km']} km of road ({h['roads_cut_fraction'] * 100:.0f}% of the network) "
        "is impassable to vehicles.",
        f"{h['shelters_viable']} shelters remain viable; {h['shelters_flooded_out']} are "
        "themselves under water and have been withdrawn.",
        f"Shelter allocation covers {h['people_sheltered']:,} people; "
        f"{h['people_unsheltered']:,} exceed available capacity.",
    ]
    if h["settlements_cut_off"]:
        lines.append(
            f"{h['settlements_cut_off']} settlements ({h['people_cut_off']:,} people) have no "
            "road route to any viable shelter and require boat or air evacuation."
        )
    if plan.is_synthetic:
        lines.append("SYNTHETIC DEMONSTRATION DATA — this does not describe a real flood.")
    return " ".join(lines)


def _instruction_text(plan: EvacuationPlan, language: str) -> str:
    parts = [translate("carry", language), translate("livestock", language)]
    if plan.boat_dispatch_list():
        parts.insert(0, translate("no_route_body", language))
    return " ".join(parts)


def cap_to_string(alert: ET.Element) -> str:
    """Serialise with an XML declaration and indentation."""
    ET.indent(alert, space="  ")
    body = ET.tostring(alert, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body


# --------------------------------------------------------------------------
# Self-validation
# --------------------------------------------------------------------------
_REQUIRED_ALERT = ("identifier", "sender", "sent", "status", "msgType", "scope")
_REQUIRED_INFO = ("category", "event", "urgency", "severity", "certainty")


def validate_cap(xml: str) -> list[str]:
    """Structural check against the CAP 1.2 requirements we can test offline.

    Not a full XSD validation — that needs the schema file and a validating
    parser. It catches the mistakes that actually happen: missing mandatory
    elements, an unclosed polygon ring, swapped coordinate order, and a
    ``status`` of Actual sneaking out of a prototype.

    Returns a list of problems; empty means it passed.
    """
    problems: list[str] = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        return [f"XML does not parse: {exc}"]

    if not root.tag.endswith("alert"):
        problems.append(f"root element is <{root.tag}>, expected <alert>")

    def find(parent, name):
        return parent.find(f"{{{CAP_NS}}}{name}")

    for name in _REQUIRED_ALERT:
        if find(root, name) is None:
            problems.append(f"missing mandatory <{name}> on <alert>")

    status = find(root, "status")
    if status is not None and status.text == "Actual":
        problems.append(
            "status is 'Actual'. A prototype must not emit a live public emergency alert."
        )

    infos = root.findall(f"{{{CAP_NS}}}info")
    if not infos:
        problems.append("no <info> block")

    for i, info in enumerate(infos):
        for name in _REQUIRED_INFO:
            if find(info, name) is None:
                problems.append(f"info[{i}]: missing mandatory <{name}>")

        for area in info.findall(f"{{{CAP_NS}}}area"):
            if find(area, "areaDesc") is None:
                problems.append(f"info[{i}]: <area> without <areaDesc>")
            poly = find(area, "polygon")
            if poly is not None and poly.text:
                pairs = poly.text.split()
                if len(pairs) < 4:
                    problems.append(f"info[{i}]: polygon has fewer than 4 points")
                elif pairs[0] != pairs[-1]:
                    problems.append(f"info[{i}]: polygon ring is not closed")
                for pair in pairs:
                    try:
                        lat, lon = (float(v) for v in pair.split(","))
                    except ValueError:
                        problems.append(f"info[{i}]: malformed coordinate '{pair}'")
                        break
                    if not (-90 <= lat <= 90):
                        problems.append(
                            f"info[{i}]: latitude {lat} out of range — coordinates are "
                            "probably in lon,lat order (CAP requires lat,lon)"
                        )
                        break
    return problems
