"""Separate FLOOD from RIVER.

A SAR water mask over Barpeta contains the Brahmaputra. The Brahmaputra is
always there. Reporting it as flood inflates flooded area, inflates exposed
population wherever the channel clips a populated pixel, and — worst — makes
the system look like it is detecting a catastrophe on a dry day.

JRC Global Surface Water (Pekel et al., Nature 2016) maps, for every 30 m
pixel on Earth, the percentage of observations between 1984 and 2021 in which
water was present. Pixels above a high occurrence threshold are permanent
water. Subtracting them leaves the flood.

This is a CORRECTNESS FIX first and a validation input second. It is also a
good answer to "how do you know you are not just detecting the river?", which
is one of the first questions a remote-sensing person will ask.

Available on Planetary Computer as ``jrc-gsw``, free and keyless.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..grid import Grid
from ..ingest.base import IngestResult, cache_path, requires, retry

logger = logging.getLogger(__name__)

__all__ = ["fetch_permanent_water", "separate_flood_from_permanent"]

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


@requires("pystac_client", "planetary_computer", "rasterio")
@retry(attempts=2, backoff_s=3.0)
def fetch_permanent_water(
    grid: Grid,
    bbox: tuple[float, float, float, float],
    occurrence_threshold: float = 80.0,
    district_id: str = "unknown",
    use_cache: bool = True,
) -> IngestResult[np.ndarray]:
    """Boolean permanent-water mask from JRC Global Surface Water occurrence.

    ``occurrence_threshold`` is the percentage of historical observations in
    which a pixel was water. 80% is deliberately high: on a braided river the
    channel MIGRATES, so a moderate threshold would mark a wide belt of
    seasonally-wet sandbar as permanent and erase real flooding over the chars
    — which is precisely where the people we care about live.
    """
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    key = f"jrc-gsw|{occurrence_threshold}|{grid.to_dict()}"
    cached = cache_path(district_id, "permanent_water", key)
    if use_cache and cached.exists():
        try:
            with np.load(cached) as npz:
                return IngestResult(
                    ok=True, data=npz["permanent"].astype(bool), source="jrc-gsw",
                    from_cache=True, detail=f"cached, occurrence >= {occurrence_threshold}%",
                )
        except Exception as exc:
            logger.warning("Permanent-water cache unreadable (%s); refetching", exc)

    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=["jrc-gsw"], bbox=list(bbox), limit=16).items())
    if not items:
        return IngestResult.failed("jrc-gsw", f"no Global Surface Water tiles cover {bbox}")

    a, b, c, d, e, f = grid.transform
    dst_transform = Affine(a, b, c, d, e, f)
    occurrence = np.zeros(grid.shape, dtype=np.float32)

    for item in items:
        asset = item.assets.get("occurrence")
        if asset is None:
            continue
        tile = np.zeros(grid.shape, dtype=np.float32)
        try:
            with rasterio.open(asset.href) as src:
                reproject(
                    source=rasterio.band(src, 1), destination=tile,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=CRS.from_string(grid.crs),
                    resampling=Resampling.max, src_nodata=255, dst_nodata=0,
                )
        except Exception as exc:
            logger.warning("JRC GSW tile %s unreadable: %s", item.id, exc)
            continue
        occurrence = np.maximum(occurrence, tile)

    permanent = occurrence >= occurrence_threshold
    if use_cache:
        try:
            np.savez_compressed(cached, permanent=permanent)
        except Exception as exc:
            logger.warning("Could not cache permanent-water mask: %s", exc)

    logger.info(
        "JRC GSW: %.2f%% of the district is permanent water (occurrence >= %.0f%%)",
        100 * permanent.mean(), occurrence_threshold,
    )
    return IngestResult(
        ok=True, data=permanent, source="jrc-gsw",
        detail=(
            f"JRC Global Surface Water 1984-2021, occurrence >= {occurrence_threshold}%, "
            f"{100 * permanent.mean():.2f}% of district"
        ),
        meta={"permanent_fraction": float(permanent.mean())},
    )


def separate_flood_from_permanent(
    water_mask: np.ndarray,
    permanent: Optional[np.ndarray],
    grid: Grid,
    dilate_cells: int = 1,
) -> dict:
    """Split a detected water mask into flood and permanent river.

    ``dilate_cells`` widens the permanent mask slightly before subtraction.
    Geolocation between JRC GSW and a Sentinel-1 scene is good but not
    pixel-exact, and without a one-cell buffer the channel edge survives
    subtraction as a persistent one-pixel ribbon of "flood" tracing the entire
    river — which looks like a bug because it is one.
    """
    water = np.asarray(water_mask) > 0

    if permanent is None:
        return {
            "flood_mask": water.astype(np.uint8),
            "permanent_mask": np.zeros_like(water, dtype=np.uint8),
            "flood_km2": float(water.sum()) * grid.cell_area_m2 / 1e6,
            "permanent_km2": 0.0,
            "total_water_km2": float(water.sum()) * grid.cell_area_m2 / 1e6,
            "separated": False,
            "caveat": (
                "Permanent water was NOT subtracted — the reported extent includes the "
                "river itself and is therefore an over-estimate of flooding."
            ),
        }

    perm = np.asarray(permanent) > 0
    if dilate_cells > 0:
        from scipy import ndimage

        size = 2 * dilate_cells + 1
        perm = ndimage.binary_dilation(perm, np.ones((size, size), dtype=bool))

    flood = water & ~perm
    cell_km2 = grid.cell_area_m2 / 1e6

    result = {
        "flood_mask": flood.astype(np.uint8),
        "permanent_mask": (water & perm).astype(np.uint8),
        "flood_km2": float(flood.sum()) * cell_km2,
        "permanent_km2": float((water & perm).sum()) * cell_km2,
        "total_water_km2": float(water.sum()) * cell_km2,
        "separated": True,
        "caveat": "",
    }
    logger.info(
        "Water separation: %.1f km2 total = %.1f km2 FLOOD + %.1f km2 permanent river",
        result["total_water_km2"], result["flood_km2"], result["permanent_km2"],
    )
    return result
