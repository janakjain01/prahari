"""End-to-end tests — the pipeline and the API, offline.

These are the tests that would have caught the two worst bugs found during
development:

  * the dashboard blanking entirely when WebGL failed, because every render
    was gated on the map loading;
  * the detector reporting MORE flooding at low water than at high, because
    tiled Otsu was splitting land against built-up area.

Both were invisible to unit tests and obvious the moment the whole chain ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from prahari.config import load_config
from prahari.decision import TravelMode, build_evacuation_plan
from prahari.demo.synth import SynthConfig, synthesise_district, synthesise_sar
from prahari.hazard.infer import detect_flood
from prahari.hazard.metrics import segmentation_metrics
from prahari.scenario import Provenance


# A coarse grid keeps this suite honest about speed. Downsample 6 over Barpeta
# is ~180 m cells: too coarse to quote, plenty to exercise every code path.
SYNTH = SynthConfig(downsample=6, n_villages=26, n_char_villages=6, n_shelters=40)


@pytest.fixture(scope="module")
def district():
    return synthesise_district(load_config("barpeta"), SYNTH)


def _plan_at(district, config, stage_m, mode=TravelMode.FOOT):
    vv, vh, truth = synthesise_sar(district, stage_m=stage_m, seed=4)
    detection = detect_flood(
        vv, vh, district.hand, already_db=True,
        max_depth_m=config.hydrology.max_depth_m, prefer="otsu",
    )
    scenario = district.with_hazard(
        detection.mask, detection.depth, detection.confidence,
        provenance=Provenance("hazard", detection.method,
                              detail=detection.method_detail, is_synthetic=True),
    )
    return scenario, build_evacuation_plan(scenario, config, mode=mode), detection, truth


class TestEndToEnd:
    def test_pipeline_produces_a_coherent_plan(self, district, config):
        scenario, plan, detection, truth = _plan_at(district, config, 3.0)

        assert detection.flooded_fraction > 0
        assert segmentation_metrics(detection.mask, truth).iou > 0.6

        headline = plan.headline()
        assert headline["needing_evacuation"] > 0
        assert headline["people_sheltered"] + headline["people_unsheltered"] == \
            plan.assignment.total_population
        assert plan.compute_seconds < 60

    def test_flood_extent_increases_monotonically_with_stage(self, district, config):
        """The regression test for the tiled-Otsu failure.

        Detected extent MUST rise with water level. When it did not, the
        system reported a worse flood at 0.4 m than at 3.0 m — a result that
        looked plausible on any single frame and was obviously wrong across
        the series.
        """
        extents = []
        for stage in (0.5, 1.5, 2.5, 3.5, 4.5):
            vv, vh, _truth = synthesise_sar(district, stage_m=stage, seed=4)
            detection = detect_flood(vv, vh, district.hand, already_db=True, prefer="otsu")
            extents.append(detection.flooded_fraction)

        assert all(b >= a - 0.01 for a, b in zip(extents, extents[1:])), (
            f"flood extent is not monotonic in river stage: {extents}"
        )

    def test_worse_flood_needs_more_evacuation(self, district, config):
        _s1, low, _d1, _t1 = _plan_at(district, config, 1.5)
        _s2, high, _d2, _t2 = _plan_at(district, config, 4.0)
        assert high.headline()["needing_evacuation"] > low.headline()["needing_evacuation"]
        assert high.roads.cut_fraction >= low.roads.cut_fraction

    def test_no_shelter_is_ever_overfilled(self, district, config):
        for stage in (1.0, 3.0, 5.0):
            _scenario, plan, _d, _t = _plan_at(district, config, stage)
            for shelter_id, load in plan.assignment.shelter_load.items():
                capacity = plan.assignment.shelter_capacity[shelter_id]
                assert load <= capacity, f"stage {stage}: {shelter_id} {load} > {capacity}"

    def test_flooded_shelters_are_withdrawn(self, district, config):
        _scenario, plan, _d, _t = _plan_at(district, config, 5.0)
        assert plan.shelters.flooded_out > 0, "a severe flood should submerge some shelters"
        for shelter in _scenario.shelters:
            if not shelter.is_viable:
                assert plan.assignment.shelter_load.get(shelter.id, 0) == 0

    def test_foot_mode_reaches_at_least_as_far_as_vehicle(self, district, config):
        """Pedestrians are blocked at a greater depth than vehicles, so the
        foot network must never be more fragmented than the vehicle one."""
        _s1, on_foot, _d1, _t1 = _plan_at(district, config, 3.0, TravelMode.FOOT)
        _s2, by_road, _d2, _t2 = _plan_at(district, config, 3.0, TravelMode.VEHICLE)
        assert on_foot.graph.removed_edges <= by_road.graph.removed_edges

    def test_plan_serialises_to_json(self, district, config):
        import json

        _scenario, plan, _d, _t = _plan_at(district, config, 3.0)
        payload = json.dumps(plan.to_dict())
        assert len(payload) > 1000
        assert json.loads(payload)["headline"]["needing_evacuation"] >= 0

    def test_scenario_roundtrips_through_disk(self, district, config, tmp_path):
        from prahari.scenario import Scenario

        scenario, _plan, _d, _t = _plan_at(district, config, 3.0)
        scenario.save(tmp_path / "scn")
        restored = Scenario.load(tmp_path / "scn")

        assert restored.grid == scenario.grid
        assert len(restored.settlements) == len(scenario.settlements)
        assert restored.roads.number_of_edges() == scenario.roads.number_of_edges()
        np.testing.assert_array_equal(restored.flood_mask, scenario.flood_mask)


class TestDisclosure:
    def test_synthetic_data_always_carries_a_banner(self, district, config):
        """The dashboard must be physically unable to present synthetic data
        as real. The banner is generated from provenance, not from a flag
        someone might forget to set."""
        from prahari.delivery.dashboard import build_dashboard_payload

        scenario, plan, _d, _t = _plan_at(district, config, 3.0)
        payload = build_dashboard_payload(scenario, plan, config, include_layers=False)

        assert payload["disclosure"]["is_synthetic"] is True
        assert any("SYNTHETIC" in note for note in payload["disclosure"]["notes"])

    def test_estimated_capacity_is_flagged(self, district, config):
        from prahari.delivery.dashboard import build_dashboard_payload

        scenario, plan, _d, _t = _plan_at(district, config, 3.0)
        payload = build_dashboard_payload(scenario, plan, config, include_layers=True)
        for feature in payload["layers"]["shelters"]["features"]:
            assert feature["properties"]["capacity_is_estimate"] is True


class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        import os

        os.environ["PRAHARI_DOWNSAMPLE"] = "6"
        from fastapi.testclient import TestClient

        from prahari.api.main import app

        return TestClient(app)

    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_districts_lists_both_configs(self, client):
        ids = {d["id"] for d in client.get("/api/districts").json()["districts"]}
        assert {"barpeta", "darbhanga"} <= ids

    def test_state_payload_is_complete(self, client):
        response = client.get("/api/state", params={"district": "barpeta", "frame": 12})
        assert response.status_code == 200
        payload = response.json()
        for key in ("plan", "layers", "overlay", "disclosure", "provenance", "thresholds"):
            assert key in payload
        assert payload["disclosure"]["is_synthetic"] is True

    def test_overlay_returns_a_png(self, client):
        response = client.get("/api/overlay.png", params={"district": "barpeta", "frame": 12})
        assert response.status_code == 200
        assert response.content[:4] == b"\x89PNG"

    def test_cap_endpoint_emits_valid_xml(self, client):
        response = client.get("/api/cap/validate", params={"district": "barpeta", "frame": 12})
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True, body["problems"]

    def test_cards_endpoint_returns_all_languages(self, client):
        response = client.get("/api/cards", params={"district": "barpeta", "frame": 12, "limit": 2})
        assert response.status_code == 200
        body = response.json()
        assert {c["language"] for c in body["cards"]} == {"en", "as", "bn", "hi"}
        assert "NOT been validated by a native speaker" in body["translation_caveat"]

    def test_unknown_language_is_a_400_with_a_useful_message(self, client):
        response = client.get("/api/cards", params={"district": "barpeta", "language": "fr"})
        assert response.status_code == 400
        assert "unsupported language" in response.json()["detail"]

    def test_dispatch_csv_downloads(self, client):
        response = client.get("/api/priorities.csv", params={"district": "barpeta", "frame": 12})
        assert response.status_code == 200
        assert response.text.splitlines()[0].startswith("rank,settlement,reason")
