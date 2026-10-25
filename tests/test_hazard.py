"""Tests for the hazard layer: HAND hydrology, Otsu detection, depth."""

from __future__ import annotations

import numpy as np
import pytest

from prahari.hazard.depth import depth_from_hand
from prahari.hazard.hand import (
    compute_hand,
    d8_receivers,
    fill_depressions,
    flow_accumulation,
    resolve_flats,
)
from prahari.hazard.metrics import compare_to_baseline, segmentation_metrics
from prahari.hazard.otsu import detect_water_otsu, otsu_threshold, to_db


class TestFillDepressions:
    def test_pit_is_filled_to_its_rim(self):
        dem = np.full((9, 9), 10.0)
        dem[4, 4] = 2.0
        filled = fill_depressions(dem)
        assert filled[4, 4] == pytest.approx(10.0)

    def test_genuine_slope_is_untouched(self):
        dem = np.tile(np.arange(9, dtype=float), (9, 1))
        filled = fill_depressions(dem)
        np.testing.assert_allclose(filled, dem)

    def test_boundary_outlet_is_preserved(self):
        """Water must still be able to leave through the edge."""
        dem = np.full((7, 7), 10.0)
        dem[3, 0] = 1.0
        filled = fill_depressions(dem)
        assert filled[3, 0] == pytest.approx(1.0)


class TestFlowRouting:
    def test_receivers_point_downhill(self):
        dem = np.tile(np.arange(10, dtype=float), (10, 1))[:, ::-1]  # falls eastward
        receivers, codes = d8_receivers(dem, cell_size=1.0)
        flat = dem.ravel()
        for i, j in enumerate(receivers):
            if j != i:
                assert flat[j] < flat[i], "receiver must be strictly lower"

    def test_sinks_point_at_themselves(self):
        dem = np.full((5, 5), 5.0)
        receivers, codes = d8_receivers(dem, cell_size=1.0)
        assert np.all(receivers == np.arange(25))
        assert np.all(codes == -1)

    def test_accumulation_conserves_cells(self):
        """Total drainage into the outlets must equal the number of cells.

        If it does not, the topological ordering is wrong and every downstream
        HAND value is unreliable.
        """
        rng = np.random.default_rng(0)
        dem = 50 - np.tile(np.arange(12, dtype=float), (12, 1)) + rng.normal(0, 0.01, (12, 12))
        filled = fill_depressions(dem)
        graded = resolve_flats(filled)
        receivers, _ = d8_receivers(graded, cell_size=1.0)
        accum = flow_accumulation(graded, receivers)

        outlets = [i for i in range(dem.size) if receivers[i] == i]
        assert sum(accum.ravel()[i] for i in outlets) == pytest.approx(dem.size)

    def test_flat_resolution_creates_drainage(self):
        """Flat terrain must acquire a usable gradient.

        The Brahmaputra floodplain is nearly flat; without this step D8 is
        undefined there and HAND is meaningless.
        """
        dem = np.full((20, 20), 10.0)
        dem[:, 0] = 9.0                       # one low edge to drain toward
        graded = resolve_flats(fill_depressions(dem))
        receivers, codes = d8_receivers(graded, cell_size=1.0)
        assert (codes >= 0).mean() > 0.5, "most flat cells should now have a downslope"


class TestHydrologicConditioning:
    """Regression tests for the worst bug found in this project.

    On the real Barpeta DEM, depression filling left 43.8% of the district as
    perfectly flat cells. Resolving those with a Euclidean distance transform
    did not work — EDT measures distance to the nearest draining cell
    GEOMETRICALLY, ignoring elevation, so the gradient often pointed uphill
    across a ridge. 15.7% of cells stayed as sinks, the flow network shattered
    into ~320,000 one-cell basins, maximum flow accumulation was 595 cells out
    of two million, no cell ever reached the channel threshold, and HAND
    silently fell back to "lowest 1% of terrain".

    The symptom that exposed it: HAND was IDENTICAL across an 80x range of
    drainage thresholds. Every depth in the system was wrong, and nothing
    crashed.
    """

    def test_conditioning_leaves_no_interior_sinks(self):
        """The guarantee Priority-Flood+epsilon must provide."""
        from prahari.hazard.hand import priority_flood_epsilon

        rng = np.random.default_rng(0)
        # A flat plain with pits and noise — the shape that broke the old code.
        dem = np.full((60, 60), 40.0)
        dem += rng.normal(0, 0.002, dem.shape)
        dem[20:30, 20:30] -= 3.0                  # a deep pit
        dem[40:50, 10:40] = 40.0                  # a perfectly flat plateau
        dem[:, 0] = 35.0                          # an outlet on one edge

        conditioned = priority_flood_epsilon(dem)
        receivers, _ = d8_receivers(conditioned, cell_size=30.0)

        is_sink = (receivers == np.arange(dem.size)).reshape(dem.shape)
        is_sink[0, :] = is_sink[-1, :] = False
        is_sink[:, 0] = is_sink[:, -1] = False
        assert is_sink.sum() == 0, f"{is_sink.sum()} interior sinks remain after conditioning"

    def test_conditioning_never_lowers_terrain(self):
        from prahari.hazard.hand import priority_flood_epsilon

        rng = np.random.default_rng(1)
        dem = 40.0 + rng.normal(0, 1.0, (40, 40))
        conditioned = priority_flood_epsilon(dem)
        assert (conditioned >= dem - 1e-9).all(), "filling must never lower the surface"

    def test_flow_network_actually_accumulates(self):
        """On a draining surface, some cell must collect a large share of flow.

        The broken version capped accumulation at 595 cells out of 2 million —
        a flow network that never joins up. Requiring a real trunk channel
        catches that directly.
        """
        from prahari.hazard.hand import priority_flood_epsilon

        rng = np.random.default_rng(2)
        h, w = 80, 80
        dem = 40.0 + 0.01 * np.arange(w)[None, :].repeat(h, 0)   # gentle slope
        dem += rng.normal(0, 0.003, dem.shape)                   # flat-making noise

        conditioned = priority_flood_epsilon(dem)
        receivers, _ = d8_receivers(conditioned, cell_size=30.0)
        accumulation = flow_accumulation(conditioned, receivers)

        assert accumulation.max() > 0.05 * dem.size, (
            f"max accumulation {accumulation.max():.0f} of {dem.size} cells — the flow "
            "network is fragmented, not draining"
        )

    def test_hand_responds_to_the_drainage_threshold(self):
        """THE test that would have caught the bug.

        A sparser channel network must push HAND UP, because cells are then
        further above their nearest drainage. When HAND is identical across
        thresholds, the threshold is not being applied at all.
        """
        rng = np.random.default_rng(3)
        h, w = 90, 90
        rows, cols = np.mgrid[0:h, 0:w].astype(float)
        dem = 40.0 + 0.02 * cols
        dem -= 5.0 * np.exp(-((rows - h / 2) ** 2) / (2 * 4.0 ** 2))   # a valley
        dem += rng.normal(0, 0.004, dem.shape)

        sparse = compute_hand(dem, cell_size=30.0, drainage_threshold_cells=2000)
        dense = compute_hand(dem, cell_size=30.0, drainage_threshold_cells=50)

        assert dense.drainage.sum() > sparse.drainage.sum()
        assert np.nanmedian(sparse.hand) > np.nanmedian(dense.hand), (
            "a sparser drainage network must raise HAND; identical values mean the "
            "threshold is being ignored and HAND has fallen back to a default"
        )


class TestHAND:
    def test_hand_is_non_negative_and_zero_in_channel(self, valley_dem):
        result = compute_hand(valley_dem, cell_size=30.0, drainage_threshold_cells=40)
        assert np.nanmin(result.hand) >= 0.0
        assert np.nanmedian(result.hand[result.drainage]) == pytest.approx(0.0, abs=0.05)

    def test_hand_increases_away_from_the_channel(self, valley_dem):
        result = compute_hand(valley_dem, cell_size=30.0, drainage_threshold_cells=40)
        h, w = valley_dem.shape
        column = result.hand[:, w // 2]
        channel_row = int(np.argmin(column))
        assert column[max(channel_row - 12, 0)] > column[max(channel_row - 3, 0)]

    def test_degenerate_threshold_falls_back_instead_of_failing(self, valley_dem):
        """An absurd threshold must still yield a defined HAND field."""
        result = compute_hand(valley_dem, cell_size=30.0, drainage_threshold_cells=10 ** 9)
        assert np.isfinite(result.hand).all()
        assert result.drainage.any()


class TestOtsu:
    def test_threshold_splits_a_clean_bimodal_histogram(self):
        values = np.concatenate([np.random.normal(-20, 1, 5000), np.random.normal(-8, 1, 5000)])
        threshold = otsu_threshold(values)
        assert -18 < threshold < -10

    def test_to_db_matches_the_definition(self):
        np.testing.assert_allclose(to_db(np.array([1.0, 0.1, 0.01])),
                                   np.array([0.0, -10.0, -20.0]), atol=1e-9)

    def test_detects_water_in_a_synthetic_scene(self):
        rng = np.random.default_rng(3)
        truth = np.zeros((128, 128), dtype=np.uint8)
        truth[40:90, 20:110] = 1
        vv = np.where(truth > 0, -20.0, -8.0) + rng.normal(0, 1.2, truth.shape)
        result = detect_water_otsu(vv, already_db=True, tile_size=64)
        assert segmentation_metrics(result.mask, truth).iou > 0.85

    def test_bright_land_contrast_does_not_become_water(self):
        """The regression test for the failure this system actually hit.

        A dry scene containing dark fields and radar-bright built-up area is
        bimodal, and naive tiled Otsu splits it — reporting a flood on a dry
        day, and MORE flooding at low water than at high. The physical
        open-water ceiling exists to stop that.
        """
        rng = np.random.default_rng(5)
        vv = np.full((128, 128), -8.0)
        vv[30:70, 30:70] = -2.0                      # built-up, radar-bright
        vv = vv + rng.normal(0, 0.8, vv.shape)
        result = detect_water_otsu(vv, already_db=True, tile_size=64)
        assert result.mask.mean() < 0.05, (
            f"{result.mask.mean():.1%} of a DRY scene was called water "
            f"(threshold {result.threshold_db:.1f} dB)"
        )

    def test_uncalibrated_input_does_not_use_physical_thresholds(self):
        """Raw GRD digital numbers run positive; a -13 dB cut is meaningless there."""
        rng = np.random.default_rng(7)
        truth = np.zeros((128, 128), dtype=np.uint8)
        truth[30:90, 30:90] = 1
        dn = np.where(truth > 0, 16.0, 24.0) + rng.normal(0, 1.0, truth.shape)
        result = detect_water_otsu(dn, already_db=True, tile_size=64, calibrated=False)
        assert result.mask.mean() > 0.05
        assert segmentation_metrics(result.mask, truth).iou > 0.5

    def test_unobserved_area_is_reported_not_assumed_dry(self):
        rng = np.random.default_rng(11)
        vv = np.full((128, 128), -8.0) + rng.normal(0, 0.5, (128, 128))
        vv[:, 64:] = np.nan                          # outside the swath
        result = detect_water_otsu(vv, already_db=True, tile_size=64)
        assert result.coverage == pytest.approx(0.5, abs=0.02)
        assert result.observed is not None
        assert not result.mask[:, 64:].any()


class TestDepth:
    def test_depth_is_zero_outside_the_flood(self):
        hand = np.tile(np.arange(20, dtype=float), (20, 1))
        mask = (hand < 5).astype(np.uint8)
        result = depth_from_hand(mask, hand)
        assert (result.depth[mask == 0] == 0).all()

    def test_depth_decreases_with_height_above_drainage(self):
        hand = np.tile(np.arange(20, dtype=float), (20, 1))
        mask = (hand < 8).astype(np.uint8)
        result = depth_from_hand(mask, hand, mode="component")
        row = result.depth[10, :8]
        assert np.all(np.diff(row) <= 1e-6), "depth must fall as HAND rises"

    def test_depth_is_clamped_to_plausible_maximum(self):
        hand = np.zeros((20, 20))
        hand[0, 0] = 500.0                      # a DEM artefact
        mask = np.ones((20, 20), dtype=np.uint8)
        result = depth_from_hand(mask, hand, max_depth_m=8.0)
        assert result.depth.max() <= 8.0

    def test_empty_mask_produces_empty_depth(self):
        hand = np.ones((10, 10))
        result = depth_from_hand(np.zeros((10, 10), dtype=np.uint8), hand)
        assert result.depth.sum() == 0
        assert result.n_components == 0


class TestMetrics:
    def test_perfect_prediction_scores_one(self):
        truth = np.zeros((10, 10), dtype=np.uint8)
        truth[2:5, 2:5] = 1
        m = segmentation_metrics(truth, truth)
        assert m.iou == 1.0 and m.f1 == 1.0

    def test_ignored_pixels_are_excluded(self):
        """Sen1Floods11 marks unlabelled pixels -1. Scoring them corrupts everything."""
        truth = np.zeros((10, 10), dtype=np.uint8)
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[0, :] = 1                                  # all wrong, but ignored
        ignore = np.zeros((10, 10), dtype=bool)
        ignore[0, :] = True
        assert segmentation_metrics(pred, truth, ignore_mask=ignore).iou == 1.0

    def test_marginal_gain_does_not_ship_the_model(self):
        baseline = segmentation_metrics(np.ones((10, 10)), np.ones((10, 10)))
        model = segmentation_metrics(np.ones((10, 10)), np.ones((10, 10)))
        verdict = compare_to_baseline(model, baseline, min_iou_gain=0.02)
        assert verdict["model_ships"] is False
        assert "does NOT ship" in verdict["verdict"]
