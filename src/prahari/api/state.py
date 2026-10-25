"""Engine state — what the API server holds between requests.

The performance shape of this system is lopsided, and the design follows it:

    STATIC  terrain, HAND, roads, population, shelters.
            Expensive (HAND is the costly part) and NEVER changes during an
            event. Built once, cached to disk, reused forever.

    DYNAMIC flood extent, depth, and the whole decision chain.
            Cheap — under two seconds at district scale — and recomputed
            every cycle.

Separating them is what makes the replay scrubber feel instantaneous on
stage. Dragging the timeline recomputes only the dynamic half, over static
layers already in memory. A demo that stalls for twenty seconds per frame is
a demo nobody watches to the end.

Thread-safety: mutations are guarded by a lock because uvicorn will happily
run handlers concurrently, and two requests racing to build the same district
would do the expensive work twice and then fight over the cache slot.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from ..config import DistrictConfig, available_districts, data_dir, load_config, models_dir
from ..decision.graph import TravelMode
from ..decision.plan import EvacuationPlan, build_evacuation_plan
from ..demo.replay import ReplayEvent, ReplayHarness, brahmaputra_stage_curve, build_replay_from_stages
from ..demo.synth import SynthConfig, synthesise_district, synthesise_sar
from ..hazard.infer import checkpoint_status, detect_flood
from ..scenario import Provenance, Scenario

logger = logging.getLogger(__name__)

__all__ = ["Engine", "get_engine", "SourceMode"]

SourceMode = Literal["synthetic", "replay", "real"]


@dataclass
class DistrictState:
    config: DistrictConfig
    base: Scenario
    replay: Optional[ReplayHarness] = None
    event: Optional[ReplayEvent] = None
    built_at: float = field(default_factory=time.time)
    build_seconds: float = 0.0
    is_real: bool = False
    build_report: dict = field(default_factory=dict)


class Engine:
    """Holds base scenarios, replay harnesses and a small plan cache."""

    def __init__(self, downsample: int = 2, cache_to_disk: bool = True) -> None:
        self.downsample = downsample
        self.cache_to_disk = cache_to_disk
        self._districts: dict[str, DistrictState] = {}
        self._real_districts: dict[str, DistrictState] = {}
        self._plan_cache: dict[tuple, tuple[Scenario, EvacuationPlan]] = {}
        self._lock = threading.RLock()
        self._checkpoint = models_dir() / "unet_sar_flood.pt"

    # -- districts ---------------------------------------------------------
    def district(self, district_id: str, real: bool = False) -> DistrictState:
        """Return the district state, building it on first use.

        ``real=True`` builds from OBSERVATIONS — Copernicus DEM, OpenStreetMap,
        WorldPop — rather than from the procedural generator. Kept as a
        separate cache from the synthetic one so a demo can flip between them
        without either rebuild evicting the other.
        """
        store = self._real_districts if real else self._districts
        with self._lock:
            state = store.get(district_id)
            if state is not None:
                return state
        # Built OUTSIDE the lock: a real build fetches over the network and can
        # take tens of seconds; holding the lock would stall every other
        # request, including the health check.
        state = self._build_real_district(district_id) if real else self._build_district(district_id)
        with self._lock:
            store.setdefault(district_id, state)
            return store[district_id]

    def _build_real_district(self, district_id: str) -> DistrictState:
        """Build a district from real observations."""
        from ..ingest import build_real_scenario

        config = load_config(district_id)
        t0 = time.time()
        logger.info("Building REAL scenario for %s from observations", district_id)

        scenario, report = build_real_scenario(config, use_cache=True)
        if scenario is None:
            raise RuntimeError(
                f"Real scenario unavailable for {district_id}: {report.get('fatal')}"
            )
        return DistrictState(
            config=config, base=scenario, replay=None, event=None,
            build_seconds=time.time() - t0, is_real=True, build_report=dict(report),
        )

    def _build_district(self, district_id: str) -> DistrictState:
        config = load_config(district_id)
        t0 = time.time()

        cache_path = data_dir() / "cache" / district_id / f"base_ds{self.downsample}"
        base: Optional[Scenario] = None

        if self.cache_to_disk and (cache_path / "scenario.json").exists():
            try:
                base = Scenario.load(cache_path)
                logger.info("Loaded cached base scenario for %s from %s", district_id, cache_path)
            except Exception as exc:
                logger.warning("Cached scenario for %s is unreadable (%s); rebuilding", district_id, exc)
                base = None

        if base is None:
            logger.info("Building base scenario for %s (downsample=%d)", district_id, self.downsample)
            base = synthesise_district(config, SynthConfig(downsample=self.downsample))
            if self.cache_to_disk:
                try:
                    base.save(cache_path)
                    logger.info("Cached base scenario to %s", cache_path)
                except Exception as exc:
                    logger.warning("Could not cache base scenario: %s", exc)

        event = build_replay_from_stages(
            event_id=f"{district_id}-synthetic-monsoon",
            name=f"{config.name} monsoon flood (synthetic)",
            district_id=district_id,
            start=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
            stages_m=brahmaputra_stage_curve(n=28, baseline_m=0.4, peak_m=4.6),
            step_hours=6.0,
            description=(
                "A synthetic monsoon hydrograph with a realistic asymmetric shape: "
                "fast rise, slow recession. Replace with a cached Sentinel-1 series "
                "for the real July 2024 event once the archive is downloaded."
            ),
            is_real_data=False,
        )
        harness = ReplayHarness(base, event, config)

        state = DistrictState(
            config=config, base=base, replay=harness, event=event,
            build_seconds=time.time() - t0,
        )
        logger.info("District %s ready in %.1fs", district_id, state.build_seconds)
        return state

    # -- plans -------------------------------------------------------------
    def plan(
        self,
        district_id: str,
        mode: TravelMode = TravelMode.FOOT,
        source: SourceMode = "replay",
        frame: int = 0,
        stage_m: Optional[float] = None,
        imminent_band_m: float = 0.0,
        use_cache: bool = True,
        date: Optional[str] = None,
        detector: str = "otsu",
    ) -> tuple[Scenario, EvacuationPlan]:
        """Return ``(scenario, plan)`` for the requested state.

        Cached on the full parameter tuple, so scrubbing back and forth over a
        replay is instant after the first pass.
        """
        key = (district_id, mode.value, source, frame, stage_m, imminent_band_m, date, detector)
        if use_cache:
            with self._lock:
                hit = self._plan_cache.get(key)
            if hit is not None:
                return hit

        if source == "real":
            state = self.district(district_id, real=True)
            scenario = self._hazard_from_observation(state, date, detector)
        else:
            state = self.district(district_id)
            if source == "replay" and state.replay is not None and stage_m is None:
                scenario = state.replay.frame(frame)
            else:
                scenario = self._hazard_at_stage(state, stage_m if stage_m is not None else 3.0)

        plan = build_evacuation_plan(
            scenario, state.config, mode=mode, imminent_band_m=imminent_band_m
        )

        with self._lock:
            self._plan_cache[key] = (scenario, plan)
            # Bounded cache: a long replay plus mode switching would otherwise
            # accumulate every frame's rasters until the process is killed.
            if len(self._plan_cache) > 64:
                oldest = next(iter(self._plan_cache))
                self._plan_cache.pop(oldest, None)
        return scenario, plan

    def _hazard_from_observation(
        self, state: DistrictState, date: Optional[str], detector: str = "otsu"
    ) -> Scenario:
        """Apply a REAL Sentinel-1 acquisition as the hazard layer.

        Permanent water is subtracted so what the map shows is flooding rather
        than the Brahmaputra — reporting the river as flood would inflate every
        downstream number and make the system look like it detects a
        catastrophe on a dry day.
        """
        from ..ingest.sentinel1 import fetch_sentinel1_scene
        from ..validate.permanent_water import fetch_permanent_water, separate_flood_from_permanent

        config = state.config
        bbox = config.grid.bbox.as_tuple()
        target = (
            datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
            if date else datetime.now(timezone.utc)
        )

        fetched = fetch_sentinel1_scene(
            state.base.grid, bbox, target_date=target,
            collection="sentinel-1-rtc", district_id=config.id, use_cache=True,
        )
        if not fetched:
            raise RuntimeError(f"No Sentinel-1 scene near {target.date()}: {fetched.reason}")
        scene = fetched.data

        # DEFAULT IS THE OTSU BASELINE ON REAL INDIAN SCENES, deliberately.
        #
        # The U-Net wins clearly on Sen1Floods11's held-out split (IoU 0.65 vs
        # 0.50), but on the real Barpeta RTC scene the two disagree by 2.5x —
        # Otsu 320 km2, U-Net 131 km2 — and we have no ground truth for Assam
        # to settle it. The training data has visibly different radiometry
        # (VV std 3.4 dB vs 6.2 dB here), so domain shift is the likely cause.
        #
        # Until that is resolved against a reference mask, we serve the
        # detector that is physically grounded, terrain-vetoed and explainable
        # on a whiteboard — and which errs toward MORE water. In an evacuation
        # a false "flooded" costs a detour; a false "dry" costs lives.
        detection = detect_flood(
            scene.vv_db, scene.vh_db, state.base.hand,
            already_db=True,
            checkpoint=str(self._checkpoint) if self._checkpoint.exists() else None,
            max_depth_m=config.hydrology.max_depth_m,
            prefer="unet" if (detector == "unet" and self._checkpoint.exists()) else "otsu",
            calibrated=scene.calibrated,
        )

        permanent = fetch_permanent_water(state.base.grid, bbox, district_id=config.id)
        separation = separate_flood_from_permanent(
            detection.mask, permanent.data if permanent else None, state.base.grid
        )
        flood_mask = separation["flood_mask"]

        scenario = state.base.with_hazard(
            flood_mask=flood_mask,
            depth=detection.depth * (flood_mask > 0),
            confidence=detection.confidence,
            timestamp=scene.acquired,
            provenance=Provenance(
                layer="hazard",
                source=detection.method,
                detail=(
                    f"{detection.method_detail} | {fetched.detail}"
                    + ("; permanent water subtracted" if separation["separated"] else "")
                ),
                is_synthetic=False,
                is_estimate=True,
            ),
        )
        scenario.meta = {
            **scenario.meta,
            "observation": {
                "item_id": scene.item_id,
                "acquired": scene.acquired,
                "coverage": round(scene.coverage, 4),
                "calibrated": scene.calibrated,
                "flood_km2": round(separation["flood_km2"], 1),
                "permanent_km2": round(separation["permanent_km2"], 1),
            },
        }
        return scenario

    def _hazard_at_stage(self, state: DistrictState, stage_m: float) -> Scenario:
        vv, vh, _truth = synthesise_sar(state.base, stage_m=stage_m, seed=11)
        detection = detect_flood(
            vv, vh, state.base.hand,
            already_db=True,
            checkpoint=str(self._checkpoint) if self._checkpoint.exists() else None,
            max_depth_m=state.config.hydrology.max_depth_m,
            prefer="unet" if self._checkpoint.exists() else "otsu",
        )
        return state.base.with_hazard(
            flood_mask=detection.mask,
            depth=detection.depth,
            confidence=detection.confidence,
            provenance=Provenance(
                layer="hazard",
                source=detection.method,
                detail=f"{detection.method_detail} | synthetic SAR at stage {stage_m:.1f} m",
                is_synthetic=True,
                is_estimate=True,
            ),
        )

    # -- introspection -----------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            loaded = {
                did: {
                    "built_at": s.built_at,
                    "build_seconds": round(s.build_seconds, 2),
                    "grid": list(s.base.grid.shape),
                    "resolution_m": s.base.grid.resolution,
                    "settlements": len(s.base.settlements),
                    "shelters": len(s.base.shelters),
                    "replay_frames": len(s.event.frames) if s.event else 0,
                }
                for did, s in self._districts.items()
            }
            cached_plans = len(self._plan_cache)

        return {
            "available_districts": available_districts(),
            "loaded_districts": loaded,
            "cached_plans": cached_plans,
            "downsample": self.downsample,
            "hazard_model": checkpoint_status(self._checkpoint),
            "data_mode": (
                "SYNTHETIC — no real observations are in use. Run "
                "`prahari ingest` with network access to switch to real data."
            ),
        }

    def invalidate(self, district_id: Optional[str] = None) -> None:
        with self._lock:
            if district_id:
                self._districts.pop(district_id, None)
                self._plan_cache = {
                    k: v for k, v in self._plan_cache.items() if k[0] != district_id
                }
            else:
                self._districts.clear()
                self._plan_cache.clear()


_ENGINE: Optional[Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine(downsample: Optional[int] = None) -> Engine:
    """Process-wide engine singleton."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            import os

            ds = downsample if downsample is not None else int(os.environ.get("PRAHARI_DOWNSAMPLE", "2"))
            _ENGINE = Engine(downsample=ds)
        return _ENGINE
