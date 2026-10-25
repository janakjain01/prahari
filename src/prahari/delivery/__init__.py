"""Delivery layer — getting the decision to the people who act on it.

Three audiences, three products, and they are genuinely different artefacts
rather than three renderings of one:

    dashboard.py    DISTRICT OFFICER / NDRF — map, blocked roads, shelter
                    load, ranked boat dispatch list
    action_card.py  CITIZEN — one screen, one destination, one route, in the
                    reader's own language
    cap.py          MACHINES — CAP 1.2 XML, the same schema SACHET uses, so
                    this plugs into the national pipeline and reaches feature
                    phones by cell broadcast

The CAP export is what answers "how does this reach someone without a
smartphone?" — the answer already exists in the architecture rather than
being a promise about future work.
"""

from .i18n import LANGUAGES, translate, language_name
from .action_card import ActionCard, build_action_card, build_stranded_card, build_all_cards
from .cap import CAPAlert, build_cap_alert, validate_cap
from .dashboard import build_dashboard_payload, build_map_layers

__all__ = [
    "LANGUAGES",
    "translate",
    "language_name",
    "ActionCard",
    "build_action_card",
    "build_stranded_card",
    "build_all_cards",
    "CAPAlert",
    "build_cap_alert",
    "validate_cap",
    "build_dashboard_payload",
    "build_map_layers",
]
