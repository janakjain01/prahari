"""Replay harness — step a flood event forward, hour by hour.

On demo day there will probably be no active flood. Replay is how the system
is shown working, and it must be labelled as replay every single time it is
shown. The banner is not a disclaimer bolted on at the end; it is part of the
product.

Two sources of frames:

    CACHED   real Sentinel-1 scenes for a historical event, pre-fetched to
             disk. This is the mode used for retrospective validation and the
             one to demo if the data is in hand.
    SYNTHETIC  frames rendered from a river-stage curve. Needs nothing.

A replay is a sequence of ``ReplayFrame`` objects. Static layers are shared
across all of them by reference, so a 48-hour replay costs 48 hazard rasters,
not 48 copies of the DEM. That is what keeps stepping instantaneous on stage —
and an instantaneous scrubber is worth more in a five-minute demo than any
extra feature.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from ..config import DistrictConfig
from ..scenario import Provenance, Scenario

logger = logging.getLogger(__name__)

__all__ = ["ReplayFrame", "ReplayEvent", "ReplayHarness", "build_replay_from_stages"]


@dataclass
class ReplayFrame:
    """One timestep."""

    index: int
    timestamp: str
    label: str
    stage_m: float
    source: str = "synthetic"
    scene_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "label": self.label,
            "stage_m": round(self.stage_m, 2),
            "source": self.source,
        }


@dataclass
class ReplayEvent:
    """A named historical (or synthesised) flood event."""

    id: str
    name: str
    district_id: str
    description: str
    frames: list[ReplayFrame]
    is_real_data: bool = False
    citation: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "district_id": self.district_id,
            "description": self.description,
            "n_frames": len(self.frames),
            "is_real_data": self.is_real_data,
            "citation": self.citation,
            "frames": [f.to_dict() for f in self.frames],
            "disclosure": (
                "Replay of a historical event using archived observations."
                if self.is_real_data
                else "SYNTHETIC replay. Rendered from a river-stage curve, not observed data. "
                     "Use only to demonstrate system behaviour, never as evidence."
            ),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def build_replay_from_stages(
    event_id: str,
    name: str,
    district_id: str,
    start: datetime,
    stages_m: Sequence[float],
    step_hours: float = 3.0,
    description: str = "",
    is_real_data: bool = False,
    citation: str = "",
) -> ReplayEvent:
    """Build an event from a river-stage curve.

    A real flood is not symmetric: it rises fast and recedes slowly. Supply a
    curve with that shape (see :func:`brahmaputra_stage_curve`) rather than a
    triangle, or the demo will show water draining as quickly as it arrived,
    which anyone who has seen a flood will notice immediately.
    """
    frames = []
    peak_index = int(np.argmax(stages_m)) if len(stages_m) else 0
    for i, stage in enumerate(stages_m):
        ts = start + timedelta(hours=step_hours * i)
        if i < peak_index:
            phase = "rising"
        elif i == peak_index:
            phase = "PEAK"
        else:
            phase = "receding"
        frames.append(
            ReplayFrame(
                index=i,
                timestamp=ts.isoformat(),
                label=f"{ts.strftime('%d %b %H:%M')} — {phase} ({stage:.1f} m)",
                stage_m=float(stage),
                source="cached" if is_real_data else "synthetic",
            )
        )
    return ReplayEvent(
        id=event_id, name=name, district_id=district_id,
        description=description, frames=frames,
        is_real_data=is_real_data, citation=citation,
    )


def brahmaputra_stage_curve(
    n: int = 32,
    baseline_m: float = 0.6,
    peak_m: float = 5.4,
    rise_fraction: float = 0.32,
) -> list[float]:
    """A stage hydrograph with realistic asymmetry.

    Rises over roughly a third of the window, recedes over the rest. Recession
    follows an exponential decay toward baseline, which is the standard shape
    of a falling limb.
    """
    n_rise = max(int(n * rise_fraction), 2)
    n_fall = max(n - n_rise, 2)

    # Rising limb: slow start, accelerating.
    rise = [
        baseline_m + (peak_m - baseline_m) * (i / n_rise) ** 1.8
        for i in range(n_rise)
    ]
    # Falling limb: exponential recession.
    fall = [
        baseline_m + (peak_m - baseline_m) * math.exp(-3.0 * i / n_fall)
        for i in range(n_fall)
    ]
    return rise + fall


class ReplayHarness:
    """Steps a :class:`ReplayEvent` over a base :class:`Scenario`."""

    def __init__(
        self,
        base: Scenario,
        event: ReplayEvent,
        config: DistrictConfig,
        detector: Optional[Callable] = None,
        seed: int = 7,
    ) -> None:
        self.base = base
        self.event = event
        self.config = config
        self.seed = seed
        self._detector = detector
        self._cache: dict[int, Scenario] = {}

    def __len__(self) -> int:
        return len(self.event.frames)

    def __iter__(self) -> Iterator[Scenario]:
        for i in range(len(self)):
            yield self.frame(i)

    def frame(self, index: int) -> Scenario:
        """Materialise the scenario at ``index``. Cached after first build."""
        if index in self._cache:
            return self._cache[index]
        if not (0 <= index < len(self.event.frames)):
            raise IndexError(f"frame {index} out of range (0..{len(self.event.frames) - 1})")

        frame = self.event.frames[index]
        scenario = self._build_frame(frame)
        self._cache[index] = scenario
        return scenario

    def _build_frame(self, frame: ReplayFrame) -> Scenario:
        from ..hazard.infer import detect_flood
        from .synth import synthesise_sar

        if frame.source == "cached" and frame.scene_path:
            vv, vh = _load_cached_scene(frame.scene_path)
            already_db = True
            source_label = f"Sentinel-1 archive ({Path(frame.scene_path).name})"
            synthetic = False
        else:
            vv, vh, _truth = synthesise_sar(self.base, frame.stage_m, seed=self.seed + frame.index)
            already_db = True
            source_label = f"synthetic SAR at stage {frame.stage_m:.1f} m"
            synthetic = True

        detector = self._detector or detect_flood
        detection = detector(
            vv, vh, self.base.hand,
            already_db=already_db,
            max_depth_m=self.config.hydrology.max_depth_m,
            prefer="otsu",
        )

        scenario = self.base.with_hazard(
            flood_mask=detection.mask,
            depth=detection.depth,
            confidence=detection.confidence,
            timestamp=frame.timestamp,
            provenance=Provenance(
                layer="hazard",
                source=detection.method,
                detail=f"{detection.method_detail} | {source_label}",
                is_synthetic=synthetic,
                is_estimate=True,
            ),
        )
        scenario.meta = {
            **scenario.meta,
            "replay": {
                "event_id": self.event.id,
                "event_name": self.event.name,
                "frame": frame.to_dict(),
                "is_real_data": self.event.is_real_data,
                "disclosure": self.event.to_dict()["disclosure"],
            },
        }
        return scenario


def _load_cached_scene(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached two-band SAR scene saved as .npz by the ingest layer."""
    with np.load(path) as npz:
        vv = npz["vv"]
        vh = npz["vh"] if "vh" in npz.files else np.zeros_like(vv)
    return vv, vh
