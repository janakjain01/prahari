"""Sentinel-1 SAR from Microsoft Planetary Computer.

This is the backbone feed, and the reason is availability rather than
preference: Planetary Computer serves Sentinel-1 through a public STAC API
with NO API KEY and NO WAITLIST. Everything else we would like — IMD, Google
Flood Hub — sits behind an approval queue that no amount of work shortens.

We prefer the ``sentinel-1-rtc`` collection over raw GRD because RTC is
Radiometrically Terrain Corrected: backscatter has already been normalised for
local incidence angle. Skipping that step over anything but perfectly flat
ground puts a slope-dependent bias straight into the water threshold. On the
Brahmaputra floodplain the difference is small; at the basin margins it is
not, and the code should not silently assume the easy case.

Why SAR at all — the answer to give a panel:

    Floods happen during the monsoon. During the monsoon India is under
    complete cloud cover, so optical satellites see nothing. SAR penetrates
    cloud. It is the only imagery that works on the day it matters.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..grid import Grid
from .base import IngestResult, cache_path, requires, retry

logger = logging.getLogger(__name__)

__all__ = ["search_sentinel1", "fetch_sentinel1_scene", "Sentinel1Scene"]

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


class Sentinel1Scene:
    """A two-band (VV, VH) SAR scene resampled onto the district grid, in dB."""

    def __init__(
        self,
        vv_db: np.ndarray,
        vh_db: Optional[np.ndarray],
        grid: Grid,
        item_id: str,
        acquired: str,
        collection: str,
        orbit: str = "",
        calibrated: bool = True,
    ) -> None:
        self.vv_db = vv_db
        self.vh_db = vh_db
        self.grid = grid
        self.item_id = item_id
        self.acquired = acquired
        self.collection = collection
        self.orbit = orbit
        self.calibrated = calibrated

    @property
    def coverage(self) -> float:
        """Fraction of the district this scene actually observed.

        A single Sentinel-1 pass is a ~250 km swath at an angle; it very often
        does NOT cover a whole district. Unobserved pixels are NaN, and they
        must never be silently counted as dry — that would under-report flood
        extent exactly where we have no information.
        """
        return float(np.isfinite(self.vv_db).mean())

    @property
    def observed_mask(self) -> np.ndarray:
        return np.isfinite(self.vv_db)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"vv": self.vv_db}
        if self.vh_db is not None:
            arrays["vh"] = self.vh_db
        np.savez_compressed(
            path, **arrays,
            meta=np.array(
                [self.item_id, self.acquired, self.collection, self.orbit], dtype=object
            ),
        )
        return path

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "acquired": self.acquired,
            "collection": self.collection,
            "orbit": self.orbit,
            "has_vh": self.vh_db is not None,
            "shape": list(self.vv_db.shape),
            "coverage": round(self.coverage, 4),
            "calibrated": self.calibrated,
        }


@requires("pystac_client", "planetary_computer")
@retry(attempts=3)
def search_sentinel1(
    bbox: tuple[float, float, float, float],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    collection: str = "sentinel-1-rtc",
    limit: int = 25,
) -> IngestResult[list[dict]]:
    """List Sentinel-1 items intersecting ``bbox`` in the date window.

    Results are newest first. For flood mapping you almost always want either
    the most recent scene (live) or the one nearest a known event date
    (validation).
    """
    import planetary_computer
    import pystac_client

    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(days=30))

    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start.date().isoformat()}/{end.date().isoformat()}",
        limit=limit,
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    listing = [
        {
            "id": it.id,
            "datetime": it.datetime.isoformat() if it.datetime else "",
            "orbit": str(it.properties.get("sat:orbit_state", "")),
            "platform": str(it.properties.get("platform", "")),
            "assets": sorted(it.assets),
        }
        for it in items
    ]

    if not listing:
        return IngestResult.failed(
            "sentinel-1",
            f"no {collection} scenes intersect {bbox} between "
            f"{start.date()} and {end.date()}. Widen the window, or try the "
            "sentinel-1-grd collection which has broader coverage.",
        )

    logger.info("Sentinel-1: %d scenes found; newest %s", len(listing), listing[0]["datetime"])
    return IngestResult(
        ok=True, data=listing, source="sentinel-1",
        detail=f"{len(listing)} {collection} scenes",
        meta={"collection": collection, "bbox": list(bbox)},
    )


@requires("pystac_client", "planetary_computer", "rasterio")
@retry(attempts=3)
def fetch_sentinel1_scene(
    grid: Grid,
    bbox: tuple[float, float, float, float],
    target_date: Optional[datetime] = None,
    window_days: int = 12,
    collection: str = "sentinel-1-rtc",
    item_id: Optional[str] = None,
    district_id: str = "unknown",
    use_cache: bool = True,
) -> IngestResult[Sentinel1Scene]:
    """Fetch one scene and resample it onto ``grid``, in decibels.

    Reads directly from the cloud-optimised GeoTIFFs over HTTP with a
    windowed, reprojecting read — only the district's pixels cross the
    network, not the whole 250 km swath. That is the difference between a
    ten-second fetch and a ten-minute one.
    """
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.warp import Resampling, reproject

    key = f"{collection}|{item_id or ''}|{target_date}|{grid.to_dict()}"
    cached = cache_path(district_id, "sentinel1", key)
    if use_cache and cached.exists():
        try:
            with np.load(cached, allow_pickle=True) as npz:
                meta = list(npz["meta"])
                scene = Sentinel1Scene(
                    npz["vv"], npz["vh"] if "vh" in npz.files else None,
                    grid, str(meta[0]), str(meta[1]), str(meta[2]),
                    str(meta[3]) if len(meta) > 3 else "",
                )
            logger.info("Sentinel-1: served %s from cache", scene.item_id)
            return IngestResult(
                ok=True, data=scene, source="sentinel-1", from_cache=True,
                detail=f"cached scene {scene.item_id} ({scene.acquired})",
            )
        except Exception as exc:
            logger.warning("Sentinel-1 cache unreadable (%s); refetching", exc)

    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)

    if item_id:
        item = catalog.get_collection(collection).get_item(item_id)
        if item is None:
            return IngestResult.failed("sentinel-1", f"item '{item_id}' not found in {collection}")
    else:
        target_date = target_date or datetime.now(timezone.utc)
        start = target_date - timedelta(days=window_days)
        end = target_date + timedelta(days=window_days)
        search = catalog.search(
            collections=[collection], bbox=list(bbox),
            datetime=f"{start.date().isoformat()}/{end.date().isoformat()}", limit=25,
        )
        items = list(search.items())
        if not items:
            return IngestResult.failed(
                "sentinel-1",
                f"no {collection} scene within ±{window_days} days of {target_date.date()} "
                f"over {bbox}",
            )
        # Nearest acquisition to the target date, not merely the newest —
        # for validation against a known event the date is the whole point.
        items.sort(key=lambda it: abs((it.datetime - target_date).total_seconds()))
        item = items[0]

    bands: dict[str, np.ndarray] = {}
    for pol in ("vv", "vh"):
        asset = item.assets.get(pol)
        if asset is None:
            continue
        try:
            bands[pol] = _read_onto_grid(asset.href, grid, rasterio, reproject, Resampling)
        except Exception as exc:
            logger.warning("Sentinel-1: could not read %s band: %s", pol, exc)

    if "vv" not in bands:
        return IngestResult.failed(
            "sentinel-1",
            f"scene {item.id} has no readable VV band (assets: {sorted(item.assets)})",
        )

    # RTC/GRD assets are linear power. Water detection works in dB.
    from ..hazard.otsu import to_db

    vv_db = to_db(bands["vv"]).astype(np.float32)
    vh_db = to_db(bands["vh"]).astype(np.float32) if "vh" in bands else None

    # CALIBRATION CHECK. The `sentinel-1-rtc` collection is radiometrically
    # terrain-corrected gamma0 in linear power, so dB lands in roughly
    # -30..+5. The raw `sentinel-1-grd` collection serves uncalibrated digital
    # numbers instead, and taking 10*log10 of those gives POSITIVE "dB" around
    # +15..+27. Those numbers look like data and are not comparable to any
    # published backscatter threshold.
    #
    # We detect the difference from the data rather than trusting the
    # collection name, and mark the scene. Downstream, an uncalibrated scene
    # is thresholded purely from its own histogram; the physical open-water
    # cut is not applied, because it would be meaningless.
    median_db = float(np.nanmedian(vv_db))
    calibrated = median_db < 0.0
    if not calibrated:
        logger.warning(
            "Scene %s appears UNCALIBRATED (median VV %.1f dB, expected negative). "
            "The %s collection serves raw digital numbers, not sigma0/gamma0. Water "
            "detection will use histogram-derived thresholds only; published "
            "backscatter cuts do not apply. Prefer collection='sentinel-1-rtc'.",
            item.id, median_db, collection,
        )

    scene = Sentinel1Scene(
        vv_db=vv_db, vh_db=vh_db, grid=grid, item_id=item.id,
        acquired=item.datetime.isoformat() if item.datetime else "",
        collection=collection,
        orbit=str(item.properties.get("sat:orbit_state", "")),
        calibrated=calibrated,
    )

    if scene.coverage < 0.999:
        logger.warning(
            "Scene %s covers only %.0f%% of the district. The remainder was NOT OBSERVED "
            "and must not be reported as dry. Fetch an adjacent-orbit scene to complete "
            "coverage, or expect the flood extent to be an under-estimate.",
            item.id, 100 * scene.coverage,
        )
    if use_cache:
        try:
            scene.save(cached)
        except Exception as exc:
            logger.warning("Could not cache Sentinel-1 scene: %s", exc)

    logger.info(
        "Sentinel-1: %s acquired %s (%s orbit), VH %s, %.0f%% district coverage, %s",
        scene.item_id, scene.acquired, scene.orbit or "unknown",
        "present" if vh_db is not None else "ABSENT",
        100 * scene.coverage,
        "calibrated" if calibrated else "UNCALIBRATED DN",
    )
    return IngestResult(
        ok=True, data=scene, source="sentinel-1",
        detail=(
            f"{collection} {item.id} acquired {scene.acquired[:19]}, "
            f"{100 * scene.coverage:.0f}% coverage"
            + ("" if calibrated else ", UNCALIBRATED digital numbers")
        ),
        meta=scene.to_dict(),
    )


def _read_onto_grid(href: str, grid: Grid, rasterio, reproject, Resampling) -> np.ndarray:
    """Windowed, reprojecting read of a COG straight onto the district grid."""
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    dst = np.full(grid.shape, np.nan, dtype=np.float32)
    a, b, c, d, e, f = grid.transform
    dst_transform = Affine(a, b, c, d, e, f)

    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=CRS.from_string(grid.crs),
            # Bilinear for continuous backscatter. Nearest would alias the
            # speckle field and make the histogram spikier than reality.
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
    return dst
