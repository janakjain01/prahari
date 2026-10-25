"""Tests for CAP export, action cards, translations, and the grid primitives.

The CAP tests are worth more than they look: element ORDER and coordinate
ORDER are both part of the specification and both are easy to get wrong in a
way that still produces well-formed XML.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from prahari.delivery.cap import CAP_NS, build_cap_alert, validate_cap
from prahari.delivery.i18n import LANGUAGES, LOCALES, translate
from prahari.grid import Grid, GridError


@pytest.fixture
def plan(toy_scenario, config):
    from prahari.decision import TravelMode, build_evacuation_plan

    return build_evacuation_plan(toy_scenario, config, mode=TravelMode.FOOT)


class TestGrid:
    def test_roundtrip_through_pixel_indices(self, small_grid):
        for row, col in ((0, 0), (10, 20), (59, 79)):
            x, y = small_grid.xy(row, col)
            r2, c2 = small_grid.rowcol(x, y)
            assert (int(r2), int(c2)) == (row, col)

    def test_xy_returns_the_pixel_centre(self, small_grid):
        x, y = small_grid.xy(0, 0)
        assert x == pytest.approx(small_grid.origin_x + 15.0)
        assert y == pytest.approx(small_grid.origin_y - 15.0)

    def test_bounds_are_consistent_with_size(self, small_grid):
        w, s, e, n = small_grid.bounds
        assert e - w == pytest.approx(small_grid.width * small_grid.resolution)
        assert n - s == pytest.approx(small_grid.height * small_grid.resolution)

    def test_from_bounds_expands_rather_than_clips(self):
        grid = Grid.from_bounds(0, 0, 95, 95, resolution=30, crs="EPSG:32646")
        w, s, e, n = grid.bounds
        assert e >= 95 and n >= 95, "the requested box must be fully contained"

    def test_shape_mismatch_is_rejected(self, small_grid):
        with pytest.raises(GridError):
            small_grid.sample(np.zeros((5, 5)), 0.0, 0.0)

    def test_out_of_bounds_sampling_returns_the_default(self, small_grid):
        raster = np.ones(small_grid.shape)
        value = small_grid.sample(raster, np.array([-1e9]), np.array([-1e9]), default=-99.0)
        assert value[0] == -99.0

    def test_serialisation_roundtrip(self, small_grid):
        assert Grid.from_dict(small_grid.to_dict()) == small_grid


class TestCAP:
    def test_emitted_cap_is_structurally_valid(self, plan, config):
        alert = build_cap_alert(plan, config)
        assert validate_cap(alert.xml) == []

    def test_identifier_has_no_forbidden_characters(self, plan, config):
        alert = build_cap_alert(plan, config)
        assert not set(alert.identifier) & set(" ,<&")

    def test_prototype_never_emits_status_actual(self, plan, config):
        alert = build_cap_alert(plan, config)
        root = ET.fromstring(alert.xml)
        assert root.find(f"{{{CAP_NS}}}status").text == "Exercise"

    def test_alert_element_order_matches_the_schema(self, plan, config):
        """CAP is an xsd:sequence. Out-of-order elements fail validators."""
        alert = build_cap_alert(plan, config)
        root = ET.fromstring(alert.xml)
        order = [child.tag.split("}")[-1] for child in root]
        expected_prefix = ["identifier", "sender", "sent", "status", "msgType", "scope"]
        assert order[: len(expected_prefix)] == expected_prefix

    def test_info_element_order_matches_the_schema(self, plan, config):
        alert = build_cap_alert(plan, config)
        root = ET.fromstring(alert.xml)
        info = root.find(f"{{{CAP_NS}}}info")
        tags = [child.tag.split("}")[-1] for child in info]

        def position(name):
            return tags.index(name) if name in tags else -1

        assert position("category") < position("event") < position("urgency")
        assert position("urgency") < position("severity") < position("certainty")
        assert position("effective") < position("expires")
        assert position("headline") < position("description") < position("instruction")

    def test_polygon_is_lat_lon_and_closed(self, plan, config):
        """CAP uses lat,lon — the OPPOSITE of GeoJSON. Swapping them moves the
        alert to another hemisphere while still validating as XML."""
        alert = build_cap_alert(plan, config)
        root = ET.fromstring(alert.xml)
        polygon = root.find(f"{{{CAP_NS}}}info/{{{CAP_NS}}}area/{{{CAP_NS}}}polygon")
        pairs = polygon.text.split()
        assert pairs[0] == pairs[-1], "ring must be closed"
        for pair in pairs:
            lat, lon = (float(v) for v in pair.split(","))
            assert -90 <= lat <= 90
            assert 60 <= lon <= 100, "expected Indian longitudes; axes may be swapped"

    def test_one_info_block_per_language(self, plan, config):
        alert = build_cap_alert(plan, config)
        root = ET.fromstring(alert.xml)
        infos = root.findall(f"{{{CAP_NS}}}info")
        assert len(infos) == len(config.alerting.languages)

    def test_validator_catches_swapped_coordinates(self):
        bad = f"""<?xml version="1.0"?>
        <alert xmlns="{CAP_NS}">
          <identifier>x</identifier><sender>s</sender><sent>2026-01-01T00:00:00+05:30</sent>
          <status>Exercise</status><msgType>Alert</msgType><scope>Public</scope>
          <info><category>Met</category><event>Flood</event><urgency>Immediate</urgency>
            <severity>Severe</severity><certainty>Likely</certainty>
            <area><areaDesc>d</areaDesc>
              <polygon>91.0,26.0 91.2,26.0 91.2,26.4 91.0,26.4 91.0,26.0</polygon>
            </area>
          </info>
        </alert>"""
        problems = validate_cap(bad)
        assert any("lon,lat" in p for p in problems)


class TestI18N:
    def test_every_language_defines_every_key(self):
        """A missing key in one language would blank a line of a safety message."""
        english_keys = set(LOCALES["en"])
        for language in LANGUAGES:
            missing = english_keys - set(LOCALES[language])
            assert not missing, f"{language} is missing {sorted(missing)}"

    def test_unknown_language_falls_back_to_english(self):
        assert translate("go_to", "xx") == LOCALES["en"]["go_to"]

    def test_placeholders_are_substituted(self):
        text = translate("distance_walk", "en", km="1.4", minutes=25)
        assert "1.4" in text and "25" in text

    def test_missing_placeholder_does_not_raise(self):
        """A formatting bug must not take down an evacuation notice."""
        assert translate("distance_walk", "en") != ""


class TestActionCards:
    def test_stranded_settlements_get_a_different_card(self, plan):
        from prahari.delivery.action_card import build_all_cards

        cards = build_all_cards(plan, languages=["en"])
        for card in cards:
            if card.kind == "stranded":
                assert "DO NOT" in card.title.upper()
                assert card.distance_line == ""

    def test_every_card_carries_a_banner(self, plan):
        """A synthetic or exercise card must never render as a real alert."""
        from prahari.delivery.action_card import build_all_cards

        for card in build_all_cards(plan, languages=["en"], limit=5):
            assert card.banner
            assert card.banner.upper() in card.to_text().upper()

    def test_cards_are_produced_in_every_language(self, plan):
        from prahari.delivery.action_card import build_all_cards

        cards = build_all_cards(plan, languages=list(LANGUAGES), limit=2)
        produced = {c.language for c in cards}
        assert produced == set(LANGUAGES)
