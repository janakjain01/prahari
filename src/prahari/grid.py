"""Metric raster grid — the spatial frame every layer agrees on.

Everything in PRAHARI (DEM, HAND, flood mask, depth, population) is a numpy
array on the SAME grid, in a projected CRS with metre units. That single
decision removes an entire class of bugs: no resampling mismatches, no
degrees-vs-metres confusion in hydrology, and spatial joins become array
indexing instead of geometry operations.

Pixel convention (this is the standard raster convention, stated explicitly
because getting it wrong silently shifts every result by half a cell):

    * ``origin`` is the OUTER top-left corner of pixel (0, 0).
    * Row index increases SOUTHWARD (y decreases).
    * Column index increases EASTWARD (x increases).
    * ``xy(row, col)`` returns the pixel CENTRE.

pyproj is imported lazily and only for lon/lat conversion, so the decision
core runs with numpy alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["Grid", "GridError"]


class GridError(ValueError):
    """Raised for malformed grid definitions or out-of-bounds access."""


@dataclass(frozen=True)
class Grid:
    """A north-up, square-pixel raster grid in a projected CRS.

    Attributes
    ----------
    origin_x, origin_y
        Outer top-left corner, in CRS units (metres).
    resolution
        Pixel size in metres. Square pixels only — we never need rotation or
        anisotropy, and forbidding them keeps the transform trivially invertible.
    width, height
        Raster size in pixels (columns, rows).
    crs
        Projected CRS identifier, e.g. ``"EPSG:32646"``.
    """

    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    crs: str

    # -- construction ------------------------------------------------------
    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise GridError(f"resolution must be positive, got {self.resolution}")
        if self.width <= 0 or self.height <= 0:
            raise GridError(f"grid must be non-empty, got {self.width}x{self.height}")

    @classmethod
    def from_bounds(
        cls,
        west: float,
        south: float,
        east: float,
        north: float,
        resolution: float,
        crs: str,
    ) -> Grid:
        """Build a grid covering a projected bounding box.

        The grid is expanded outward (ceil) so the box is fully contained —
        never clipped. Losing a strip of floodplain to a rounding-down is
        exactly the kind of quiet error that ruins a validation run.
        """
        if east <= west or north <= south:
            raise GridError(f"degenerate bounds ({west}, {south}, {east}, {north})")
        width = int(math.ceil((east - west) / resolution))
        height = int(math.ceil((north - south) / resolution))
        return cls(
            origin_x=west,
            origin_y=south + height * resolution,
            resolution=resolution,
            width=width,
            height=height,
            crs=crs,
        )

    @classmethod
    def from_geographic_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        resolution_m: float,
        crs: str,
        geographic_crs: str = "EPSG:4326",
    ) -> Grid:
        """Build a metric grid from a lon/lat bounding box.

        Requires pyproj. All four corners are projected and the metric
        envelope taken, because a lon/lat rectangle is not a rectangle after
        projection — using only two corners would clip the edges.
        """
        transformer = _transformer(geographic_crs, crs)
        west, south, east, north = bbox
        corners = [(west, south), (west, north), (east, south), (east, north)]
        xs, ys = zip(*(transformer.transform(lon, lat) for lon, lat in corners))
        return cls.from_bounds(min(xs), min(ys), max(xs), max(ys), resolution_m, crs)

    # -- geometry ----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Numpy array shape: ``(height, width)`` i.e. ``(rows, cols)``."""
        return (self.height, self.width)

    @property
    def size(self) -> int:
        return self.width * self.height

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` in CRS units."""
        return (
            self.origin_x,
            self.origin_y - self.height * self.resolution,
            self.origin_x + self.width * self.resolution,
            self.origin_y,
        )

    @property
    def cell_area_m2(self) -> float:
        return self.resolution * self.resolution

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL-style affine coefficients ``(a, b, c, d, e, f)``.

        Maps ``(col, row) -> (x, y)`` of the pixel's top-left corner:
        ``x = a*col + b*row + c``, ``y = d*col + e*row + f``.
        Handed to rasterio/GDAL on write.
        """
        return (self.resolution, 0.0, self.origin_x, 0.0, -self.resolution, self.origin_y)

    # -- coordinate conversion --------------------------------------------
    def xy(self, row: np.ndarray | int, col: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """Pixel indices -> CRS coordinates of the pixel CENTRE."""
        r = np.asarray(row, dtype=np.float64)
        c = np.asarray(col, dtype=np.float64)
        x = self.origin_x + (c + 0.5) * self.resolution
        y = self.origin_y - (r + 0.5) * self.resolution
        return x, y

    def rowcol(
        self, x: np.ndarray | float, y: np.ndarray | float, clip: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """CRS coordinates -> integer pixel indices (floor).

        With ``clip=True`` out-of-bounds coordinates are snapped to the nearest
        valid pixel instead of returning indices outside the array. Use it when
        placing a point feature whose surveyed coordinate sits a few metres
        outside the AOI; do NOT use it when you need to know a point is absent.
        """
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        col = np.floor((xa - self.origin_x) / self.resolution).astype(np.int64)
        row = np.floor((self.origin_y - ya) / self.resolution).astype(np.int64)
        if clip:
            col = np.clip(col, 0, self.width - 1)
            row = np.clip(row, 0, self.height - 1)
        return row, col

    def contains(self, row: np.ndarray | int, col: np.ndarray | int) -> np.ndarray:
        r = np.asarray(row)
        c = np.asarray(col)
        return (r >= 0) & (r < self.height) & (c >= 0) & (c < self.width)

    def to_geographic(
        self, x: np.ndarray | float, y: np.ndarray | float, geographic_crs: str = "EPSG:4326"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Projected coordinates -> (lon, lat). Requires pyproj."""
        transformer = _transformer(self.crs, geographic_crs)
        lon, lat = transformer.transform(np.asarray(x), np.asarray(y))
        return np.asarray(lon), np.asarray(lat)

    def from_geographic(
        self, lon: np.ndarray | float, lat: np.ndarray | float, geographic_crs: str = "EPSG:4326"
    ) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) -> projected coordinates. Requires pyproj."""
        transformer = _transformer(geographic_crs, self.crs)
        x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x), np.asarray(y)

    def geographic_bounds(self, geographic_crs: str = "EPSG:4326") -> tuple[float, float, float, float]:
        """Envelope in lon/lat, for map fitting and CAP polygons."""
        w, s, e, n = self.bounds
        xs = np.array([w, w, e, e])
        ys = np.array([s, n, s, n])
        lon, lat = self.to_geographic(xs, ys, geographic_crs)
        return (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))

    # -- sampling ----------------------------------------------------------
    def sample(
        self,
        raster: np.ndarray,
        x: np.ndarray | float,
        y: np.ndarray | float,
        default: float = float("nan"),
    ) -> np.ndarray:
        """Nearest-neighbour sample of ``raster`` at projected coordinates.

        Nearest-neighbour, not bilinear, and deliberately so: these rasters
        carry categorical or threshold-sensitive values (flood mask, depth
        band). Interpolating a flood mask invents half-flooded pixels.
        """
        self._check_raster(raster)
        row, col = self.rowcol(x, y)
        inside = self.contains(row, col)
        out = np.full(np.shape(row), default, dtype=np.float64)
        if np.any(inside):
            out[inside] = raster[row[inside], col[inside]]
        return out

    def _check_raster(self, raster: np.ndarray) -> None:
        if raster.shape[:2] != self.shape:
            raise GridError(
                f"raster shape {raster.shape[:2]} does not match grid {self.shape}. "
                "Every layer must live on the same grid — resample at ingest, not here."
            )

    # -- helpers -----------------------------------------------------------
    def coordinate_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Full ``(X, Y)`` centre-coordinate arrays, shape ``(height, width)``."""
        cols = np.arange(self.width)
        rows = np.arange(self.height)
        x = self.origin_x + (cols + 0.5) * self.resolution
        y = self.origin_y - (rows + 0.5) * self.resolution
        return np.meshgrid(x, y)

    def empty(self, dtype: np.dtype | type = np.float32, fill: float = 0.0) -> np.ndarray:
        return np.full(self.shape, fill, dtype=dtype)

    def polygon_geographic(self, geographic_crs: str = "EPSG:4326") -> list[tuple[float, float]]:
        """AOI outline as a closed lon/lat ring, for CAP ``<polygon>`` and GeoJSON."""
        w, s, e, n = self.bounds
        ring = [(w, n), (e, n), (e, s), (w, s), (w, n)]
        xs, ys = zip(*ring)
        lon, lat = self.to_geographic(np.array(xs), np.array(ys), geographic_crs)
        return [(float(a), float(b)) for a, b in zip(lon, lat)]

    def to_dict(self) -> dict:
        return {
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Grid:
        return cls(
            origin_x=float(d["origin_x"]),
            origin_y=float(d["origin_y"]),
            resolution=float(d["resolution"]),
            width=int(d["width"]),
            height=int(d["height"]),
            crs=str(d["crs"]),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        w, s, e, n = self.bounds
        return (
            f"Grid({self.width}x{self.height} @ {self.resolution:g} m, {self.crs}, "
            f"bounds=({w:.0f}, {s:.0f}, {e:.0f}, {n:.0f}))"
        )


# --------------------------------------------------------------------------
# pyproj is optional and cached — building a Transformer is expensive and we
# do it in inner loops otherwise.
# --------------------------------------------------------------------------
_TRANSFORMER_CACHE: dict[tuple[str, str], object] = {}


def _transformer(src: str, dst: str):
    key = (src, dst)
    cached = _TRANSFORMER_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GridError(
            "pyproj is required for coordinate conversion. Install the geo extra: "
            "pip install 'prahari[geo]'. (The decision core itself does not need it.)"
        ) from exc
    t = Transformer.from_crs(src, dst, always_xy=True)
    _TRANSFORMER_CACHE[key] = t
    return t


def grid_from_config(cfg) -> Grid:
    """Build the district grid from a :class:`~prahari.config.DistrictConfig`."""
    return Grid.from_geographic_bbox(
        bbox=cfg.grid.bbox.as_tuple(),
        resolution_m=cfg.grid.resolution_m,
        crs=cfg.grid.crs,
        geographic_crs=cfg.grid.geographic_crs,
    )
