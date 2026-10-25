"""Population — WorldPop 100 m constrained rasters.

WorldPop disaggregates official census totals onto a 100 m grid using
built-up area and other covariates. It is the right product for "roughly how
many people are in this square kilometre" and it is the wrong product for any
statement about an individual household.

Two things follow, and the second is the one that matters operationally:

  1. Reporting is aggregated to 1 km cells, not shown per 30 m pixel, so the
     dashboard never implies a precision the data does not have.
  2. Population is a RESIDENT count from a census year, not a live headcount.
     During a monsoon flood, seasonal labour movement and prior self-evacuation
     both mean the people actually present differ from the raster. We say so
     rather than presenting the figure as a census of who is standing there
     tonight.

Resampling note: population is an EXTENSIVE quantity — people per pixel, not a
density. Going from 100 m to 30 m with bilinear interpolation would multiply
the district's population by roughly eleven. We therefore resample by SUM,
preserving the total, and assert that the total survives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..grid import Grid
from .base import IngestResult, cache_path, requires, retry

logger = logging.getLogger(__name__)

__all__ = ["fetch_worldpop", "WORLDPOP_URL_TEMPLATE"]

# WorldPop's UN-adjusted, constrained country rasters.
WORLDPOP_URL_TEMPLATE = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
    "{year}/BSGM/{iso3}/{iso3_lower}_ppp_{year}_UNadj_constrained.tif"
)


@requires("rasterio")
@retry(attempts=2, backoff_s=3.0)
def fetch_worldpop(
    grid: Grid,
    bbox: tuple[float, float, float, float],
    year: int = 2020,
    iso3: str = "IND",
    district_id: str = "unknown",
    url: Optional[str] = None,
    use_cache: bool = True,
    allow_full_download: bool = True,
    timeout_s: float = 90.0,
) -> IngestResult[np.ndarray]:
    """Fetch population-per-pixel onto the district grid, conserving the total.

    Two access paths, tried in order:

      1. WINDOWED HTTP READ. Reads only the district's pixels out of the
         national raster. Fast when it works.
      2. FULL DOWNLOAD TO CACHE. WorldPop's server advertises
         ``Accept-Ranges: bytes`` but in practice stalls on range GETs often
         enough that path 1 cannot be relied on. When it fails we stream the
         whole country file (~490 MB for India) into the cache once and read
         it locally from then on.

    Path 2 is deliberately opt-out rather than opt-in: a half-gigabyte
    one-time download during preparation is a far better trade than a
    dashboard that reports a flat placeholder density on demo day. Do it once,
    before the event, and the file is on the laptop forever.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds

    key = f"worldpop|{year}|{iso3}|{grid.to_dict()}"
    cached = cache_path(district_id, "population", key)
    if use_cache and cached.exists():
        try:
            with np.load(cached) as npz:
                return IngestResult(
                    ok=True, data=npz["population"], source="worldpop",
                    from_cache=True, detail=f"cached WorldPop {year} {iso3}",
                )
        except Exception as exc:
            logger.warning("WorldPop cache unreadable (%s); refetching", exc)

    href = url or WORLDPOP_URL_TEMPLATE.format(year=year, iso3=iso3, iso3_lower=iso3.lower())

    # If the country raster was already downloaded — for this district or any
    # other — use it. It is the same national file, so re-fetching it over the
    # network for a second district is pure waste, and worse, a windowed HTTP
    # read can fail while a perfectly good 490 MB copy sits on disk. This is
    # checked BEFORE the network path, and independently of
    # ``allow_full_download``, which governs whether we may START a download,
    # not whether we may use one that already finished.
    local = _local_country_raster(href)
    if local is not None:
        logger.info("WorldPop: using the local country raster at %s", local)
        href = str(local)

    dst = np.zeros(grid.shape, dtype=np.float32)
    a, b, c, d, e, f = grid.transform
    dst_transform = Affine(a, b, c, d, e, f)

    def _read_from(source_path: str) -> float:
        # NOTE: do NOT set CPL_VSIL_CURL_USE_HEAD=NO. GDAL discovers HTTP range
        # support from the HEAD response; suppressing HEAD makes it conclude
        # "Range downloading not supported by this server" outright.
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY="2",
            GDAL_HTTP_RETRY_DELAY="2",
            GDAL_HTTP_TIMEOUT=str(int(timeout_s)),
        ):
            with rasterio.open(source_path) as src:
                window_bounds = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
                window = src.window(*window_bounds)
                source = src.read(1, window=window, boundless=True, fill_value=0)
                source = np.where(source < 0, 0.0, source).astype(np.float64)

                reproject(
                    source=source,
                    destination=dst,
                    src_transform=src.window_transform(window),
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=CRS.from_string(grid.crs),
                    # SUM, not bilinear. Population is people-per-pixel; any
                    # averaging or interpolation silently rescales the entire
                    # district population by the cell-area ratio.
                    resampling=Resampling.sum,
                    src_nodata=0,
                    dst_nodata=0,
                )
                return float(source.sum())

    try:
        source_total = _read_from(href)
    except Exception as exc:
        if not allow_full_download:
            return IngestResult.failed(
                "worldpop",
                f"windowed read failed ({exc}) and allow_full_download is off",
            )
        logger.warning(
            "WorldPop windowed read failed (%s). Falling back to a one-time full "
            "download of the country raster.", exc,
        )
        local = _download_country_raster(href, district_id)
        if local is None:
            return IngestResult.failed(
                "worldpop",
                f"windowed read failed ({exc}) and the full download did not complete. "
                "Run `python -m prahari.cli fetch-population` on a better connection, "
                "or supply a local GeoTIFF with --population-file.",
            )
        source_total = _read_from(str(local))

    total = float(dst.sum())
    if total <= 0:
        return IngestResult.failed(
            "worldpop", f"WorldPop window over {bbox} summed to zero people"
        )

    # Resampling.sum is not exactly conservative at the edges; if the drift is
    # material, rescale so the district total matches the source window.
    if source_total > 0:
        drift = abs(total - source_total) / source_total
        if drift > 0.02:
            logger.info(
                "WorldPop: resampling drifted %.1f%% from the source total; rescaling to conserve",
                100 * drift,
            )
            dst = (dst * (source_total / total)).astype(np.float32)
            total = float(dst.sum())

    if use_cache:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, population=dst)
        except Exception as exc:
            logger.warning("Could not cache WorldPop: %s", exc)

    logger.info("WorldPop %d: %.0f people across the district window", year, total)
    _ = source_total  # retained above for the conservation check
    return IngestResult(
        ok=True, data=dst, source="worldpop",
        detail=(
            f"WorldPop {year} UN-adjusted constrained, {total:,.0f} residents. "
            "Modelled census disaggregation, not a live headcount."
        ),
        meta={"total_population": total, "year": year},
    )


def _local_country_raster(href: str) -> Optional[Path]:
    """Return the already-downloaded country raster for this URL, if present."""
    from ..config import data_dir

    candidate = data_dir() / "raw" / "worldpop" / href.rsplit("/", 1)[-1]
    if candidate.exists() and candidate.stat().st_size > 1_000_000:
        return candidate
    return None


def _download_country_raster(href: str, district_id: str) -> Optional["Path"]:
    """Stream the national raster to the cache. Resumable, once.

    Kept as a plain streaming download rather than a range-based one precisely
    because ranges are what failed. Partial files are written to a ``.part``
    path and only renamed on success, so an interrupted download never leaves
    a truncated GeoTIFF that later reads as corrupt data.
    """
    import urllib.request
    from pathlib import Path

    from ..config import data_dir

    name = href.rsplit("/", 1)[-1]
    target = data_dir() / "raw" / "worldpop" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        logger.info("WorldPop: using previously downloaded %s", target)
        return target

    part = target.with_suffix(target.suffix + ".part")
    logger.warning(
        "Downloading %s — this is a LARGE one-time download (hundreds of MB). "
        "Do this before demo day, not during it.", name,
    )
    try:
        request = urllib.request.Request(href, headers={"User-Agent": "PRAHARI/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as fh:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last_report = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total and done - last_report > (1 << 26):   # every 64 MB
                    last_report = done
                    logger.info("  WorldPop download %.0f%% (%.0f/%.0f MB)",
                                100 * done / total, done / 1e6, total / 1e6)
        part.replace(target)
        logger.info("WorldPop: downloaded %s (%.0f MB)", target, target.stat().st_size / 1e6)
        return target
    except Exception as exc:
        logger.error("WorldPop download failed: %s", exc)
        part.unlink(missing_ok=True)
        return None
