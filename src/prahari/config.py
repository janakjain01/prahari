"""Typed configuration loaded from ``config/<district>.yaml``.

Design rule: no magic numbers anywhere else in the codebase. Every threshold a
judge might question — 0.30 m road blocking, 3.5 m2 per person, 90 minutes to
shelter — lives here, in one file, with a citation next to it. When someone
asks "where does that number come from?", we open one file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
def project_root() -> Path:
    """Repository root, resolved from this file's location.

    Works whether the package is installed (`pip install -e .`) or run
    straight from a checkout, which matters because on demo day someone will
    inevitably run it the "wrong" way.
    """
    env = os.environ.get("PRAHARI_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "config"


def data_dir() -> Path:
    d = Path(os.environ.get("PRAHARI_DATA", project_root() / "data"))
    return d.resolve()


def models_dir() -> Path:
    d = Path(os.environ.get("PRAHARI_MODELS", project_root() / "models"))
    return d.resolve()


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
class BBox(BaseModel):
    """Geographic bounding box in WGS84 degrees."""

    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if self.west >= self.east:
            raise ValueError(f"west ({self.west}) must be < east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"south ({self.south}) must be < north ({self.north})")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)


class DistrictInfo(BaseModel):
    id: str
    name: str
    state: str
    country: str = "IN"
    basin: str = ""
    lgd_code: str = ""
    census_code: str = ""


class GridConfig(BaseModel):
    crs: str
    geographic_crs: str = "EPSG:4326"
    resolution_m: float = Field(default=30.0, gt=0)
    bbox: BBox


class HydrologyConfig(BaseModel):
    """Hydrology parameters.

    The channel-initiation threshold is expressed as CONTRIBUTING AREA IN KM2,
    not in cells. Cells are the wrong unit: the same "1000 cells" means 0.9 km2
    on a 30 m grid and 8.1 km2 on a 90 m grid, so a threshold tuned at one
    resolution silently changes meaning at another — and HAND, and therefore
    every depth in the system, moves with it. Area is resolution-independent
    and is what the hydrology literature actually specifies.
    """

    drainage_threshold_km2: float = Field(default=50.0, gt=0)
    fill_depressions: bool = True
    max_depth_m: float = Field(default=8.0, gt=0)

    def threshold_cells(self, resolution_m: float) -> int:
        """Convert the area threshold to a cell count for this grid."""
        cell_km2 = (resolution_m * resolution_m) / 1e6
        return max(int(round(self.drainage_threshold_km2 / cell_km2)), 1)


class PassabilityConfig(BaseModel):
    """Depth thresholds in metres. See docs/REFERENCES.md for provenance."""

    clear_m: float = 0.15
    caution_m: float = 0.30
    vehicle_block_m: float = 0.30
    life_threatening_m: float = 0.60
    caution_speed_factor: float = Field(default=0.4, gt=0, le=1.0)

    @model_validator(mode="after")
    def _monotonic(self) -> PassabilityConfig:
        if not (self.clear_m <= self.caution_m <= self.life_threatening_m):
            raise ValueError(
                "passability thresholds must be non-decreasing: "
                f"clear={self.clear_m} caution={self.caution_m} "
                f"life_threatening={self.life_threatening_m}"
            )
        return self


class EvacuationConfig(BaseModel):
    walking_speed_kmph: float = Field(default=4.0, gt=0)
    vehicle_speed_kmph: float = Field(default=25.0, gt=0)
    max_travel_minutes: float = Field(default=90.0, gt=0)
    sqm_per_person: float = Field(default=3.5, gt=0)
    default_shelter_capacity: int = Field(default=200, gt=0)
    capacity_reserve_fraction: float = Field(default=0.10, ge=0, lt=1)


class CAPConfig(BaseModel):
    sender: str = "prahari@ndma-demo.in"
    sender_name: str = "PRAHARI Decision Layer"
    scope: Literal["Public", "Restricted", "Private"] = "Public"
    # A prototype must NEVER emit status=Actual. Emitting a live-looking public
    # emergency alert is both a legal and an ethical problem. Enforced below.
    status: Literal["Actual", "Exercise", "System", "Test", "Draft"] = "Exercise"
    urgency: Literal["Immediate", "Expected", "Future", "Past", "Unknown"] = "Immediate"
    severity: Literal["Extreme", "Severe", "Moderate", "Minor", "Unknown"] = "Severe"
    certainty: Literal["Observed", "Likely", "Possible", "Unlikely", "Unknown"] = "Likely"

    @field_validator("status")
    @classmethod
    def _no_actual_without_optin(cls, v: str) -> str:
        if v == "Actual" and os.environ.get("PRAHARI_ALLOW_ACTUAL_CAP") != "1":
            raise ValueError(
                "CAP status='Actual' refused. This is a prototype; emitting a "
                "real-looking public emergency alert is not something to do by "
                "accident. Set PRAHARI_ALLOW_ACTUAL_CAP=1 only in an authorised "
                "operational deployment."
            )
        return v


class AlertingConfig(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["en"])
    cap: CAPConfig = Field(default_factory=CAPConfig)


class SourcesConfig(BaseModel):
    """Loose by design — source blocks vary per provider and we do not want a
    schema change every time a feed adds a field."""

    model_config = {"extra": "allow"}

    dem: dict[str, Any] = Field(default_factory=dict)
    sar: dict[str, Any] = Field(default_factory=dict)
    population: dict[str, Any] = Field(default_factory=dict)
    osm: dict[str, Any] = Field(default_factory=dict)
    precipitation: dict[str, Any] = Field(default_factory=dict)
    forecast: dict[str, Any] = Field(default_factory=dict)


class DistrictConfig(BaseModel):
    """The whole configuration for one district."""

    district: DistrictInfo
    grid: GridConfig
    hydrology: HydrologyConfig = Field(default_factory=HydrologyConfig)
    passability: PassabilityConfig = Field(default_factory=PassabilityConfig)
    evacuation: EvacuationConfig = Field(default_factory=EvacuationConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)

    # convenience passthroughs -------------------------------------------------
    @property
    def id(self) -> str:
        return self.district.id

    @property
    def name(self) -> str:
        return self.district.name

    @property
    def usable_capacity_factor(self) -> float:
        return 1.0 - self.evacuation.capacity_reserve_fraction

    def scenario_dir(self) -> Path:
        return data_dir() / "scenarios" / self.district.id

    def cache_dir(self) -> Path:
        return data_dir() / "cache" / self.district.id

    def raw_dir(self) -> Path:
        return data_dir() / "raw" / self.district.id


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def load_config(district: str = "barpeta") -> DistrictConfig:
    """Load and validate ``config/<district>.yaml``.

    Cached: configuration is read once per process. Call ``load_config.cache_clear()``
    in tests that mutate YAML on disk.
    """
    path = config_dir() / f"{district}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in config_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"No config for district '{district}' at {path}. Available: {available or '(none)'}"
        )
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return DistrictConfig.model_validate(raw)


def available_districts() -> list[str]:
    return sorted(p.stem for p in config_dir().glob("*.yaml"))
