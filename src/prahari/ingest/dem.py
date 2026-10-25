"""Elevation — Copernicus DEM GLO-30 via Planetary Computer.

Copernicus GLO-30 rather than SRTM: it is newer, void-filled, and generally
better behaved over flat deltaic terrain, which is exactly the ground this
project cares about.

The honest limitation, and it belongs in the pitch rather than in a footnote:
a global 30 m DEM does NOT resolve embankments. An Assam embankment is a few
metres wide and a couple of metres high — sub-pixel in both dimensions. So
PRAHARI cannot see an intact embankment holding water back, and cannot see a
breach opening. In a district where breached embankments are a primary flood
mechanism, that is a real constraint, not a nitpick.

What we do about it, rather than shrug: embankment centrelines exist in OSM
(``man_made=dyke``, ``embankment=yes``) and can be burned into the DEM as a
raised barrier. ``burn_embankments()`` does that, and the operator can see
that it was applied.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..grid import Grid
from .base import IngestResult, cache_path, requires, retry

logger = logging.getLogger(__name__)

__all__ = ["fetch_dem", "burn_embankments"]

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


@requires("pystac_client", "planetary_computer", "rasterio")
@retry(attempts=3)
def fetch_dem(
    grid: Grid,
    bbox: tuple[float, float, float, float],
    collection: str = "cop-dem-glo-30",
    district_id: str = "unknown",
    use_cache: bool = True,
) -> IngestResult[np.ndarray]:
    """Fetch and mosaic elevation onto the district grid, in metres."""
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    key = f"{collection}|{grid.to_dict()}"
    cached = cache_path(district_id, "dem", key)
    if use_cache and cached.exists():
        try:
            with np.load(cached) as npz:
                return IngestResult(
                    ok=True, data=npz["dem"], source="copernicus-dem",
                    from_cache=True, detail=f"cached {collection}",
                )
        except Exception as exc:
            logger.warning("DEM cache unreadable (%s); refetching", exc)

    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=[collection], bbox=list(bbox), limit=64).items())
    if not items:
        return IngestResult.failed("copernicus-dem", f"no {collection} tiles cover {bbox}")

    a, b, c, d, e, f = grid.transform
    dst_transform = Affine(a, b, c, d, e, f)
    mosaic = np.full(grid.shape, np.nan, dtype=np.float32)

    for item in items:
        asset = item.assets.get("data") or next(iter(item.assets.values()), None)
        if asset is None:
            continue
        tile = np.full(grid.shape, np.nan, dtype=np.float32)
        try:
            with rasterio.open(asset.href) as src:
                reproject(
                    source=rasterio.band(src, 1), destination=tile,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=CRS.from_string(grid.crs),
                    resampling=Resampling.bilinear, dst_nodata=np.nan,
                )
        except Exception as exc:
            logger.warning("DEM tile %s unreadable: %s", item.id, exc)
            continue
        # Later tiles fill only what earlier ones left empty, so overlapping
        # tile edges do not overwrite good data with edge artefacts.
        fill = np.isnan(mosaic) & ~np.isnan(tile)
        mosaic[fill] = tile[fill]

    coverage = float(np.isfinite(mosaic).mean())
    if coverage < 0.5:
        return IngestResult.failed(
            "copernicus-dem",
            f"only {coverage:.0%} of the grid has elevation data from {len(items)} tiles",
        )
    if coverage < 0.999:
        # Small voids: fill from the nearest valid cell. Leaving NaN would
        # propagate through HAND and silently blank part of the district.
        from scipy import ndimage

        holes = ~np.isfinite(mosaic)
        _d, (ri, ci) = ndimage.distance_transform_edt(holes, return_indices=True)
        mosaic[holes] = mosaic[ri[holes], ci[holes]]
        logger.info("DEM: filled %d void cells by nearest-neighbour", int(holes.sum()))

    if use_cache:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, dem=mosaic)
        except Exception as exc:
            logger.warning("Could not cache DEM: %s", exc)

    logger.info(
        "DEM: %s from %d tiles, %.1f–%.1f m",
        collection, len(items), float(np.nanmin(mosaic)), float(np.nanmax(mosaic)),
    )
    return IngestResult(
        ok=True, data=mosaic, source="copernicus-dem",
        detail=f"{collection}, {len(items)} tiles, {coverage:.1%} coverage",
        meta={"tiles": len(items), "coverage": coverage},
    )


def burn_embankments(
    dem: np.ndarray,
    grid: Grid,
    embankments: Any,
    height_m: float = 2.5,
    width_cells: int = 1,
) -> np.ndarray:
    """Raise DEM cells along embankment centrelines.

    A 30 m DEM cannot resolve a 4 m wide embankment, so hydrology run on the
    raw DEM lets water flow straight through structures that are, in reality,
    the only thing keeping it out. Burning the known lines in restores the
    barrier.

    ``embankments`` is a GeoDataFrame of LineStrings in the grid CRS —
    typically OSM ``man_made=dyke`` or ``embankment=yes``.

    STATE THIS WHEN ASKED. It is an intervention on the input data, it is
    visible in provenance, and its height is an assumption.
    """
    out = np.array(dem, dtype=np.float64, copy=True)
    if embankments is None or len(embankments) == 0:
        return out

    # Build a mask first, then apply the rise ONCE. Raising in the inner loop
    # would stack an extra height_m every time two densified sample points
    # landed in the same cell, which on a long dyke means a wall tens of
    # metres tall and a completely wrong flow network.
    mask = np.zeros(grid.shape, dtype=bool)

    for geom in embankments.geometry:
        coords = _line_coords(geom)
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            n = max(int(np.hypot(x1 - x0, y1 - y0) / (grid.resolution / 2)), 2)
            ts = np.linspace(0.0, 1.0, n)
            xs = x0 + ts * (x1 - x0)
            ys = y0 + ts * (y1 - y0)
            rows, cols = grid.rowcol(xs, ys)
            inside = grid.contains(rows, cols)
            mask[rows[inside], cols[inside]] = True

    if width_cells > 0 and mask.any():
        from scipy import ndimage

        size = 2 * width_cells + 1
        mask = ndimage.binary_dilation(mask, np.ones((size, size), dtype=bool))

    out[mask] = out[mask] + height_m
    logger.info(
        "Burned %d embankment cells at +%.1f m into the DEM (%.2f%% of the grid)",
        int(mask.sum()), height_m, 100.0 * mask.mean(),
    )
    return out


def _line_coords(geom) -> list[tuple[float, float]]:
    if hasattr(geom, "coords"):
        return [(float(x), float(y)) for x, y in geom.coords]
    if hasattr(geom, "geoms"):
        out: list[tuple[float, float]] = []
        for part in geom.geoms:
            out.extend(_line_coords(part))
        return out
    return []
