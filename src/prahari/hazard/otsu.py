"""Classical SAR water detection — the baseline the U-Net has to beat.

Physics, in one line: open water is a specular reflector, so it bounces radar
energy away from the sensor and appears very dark in a SAR image. Land is
rough at C-band wavelengths, scatters diffusely, and appears bright. Water
detection is therefore a thresholding problem on the backscatter histogram.

Why SAR at all, and this is the answer to give a panel:

    During the monsoon — the only time floods actually happen — India is under
    complete cloud cover. Optical satellites see nothing. Synthetic Aperture
    Radar penetrates cloud. It is the only imagery that works on the day it
    matters.

Two details separate this from a naive global Otsu, and both are real
literature, not embellishment:

    1. TILE SELECTION. Over a whole district the flooded fraction is usually
       small, so the global histogram is unimodal and Otsu returns nonsense.
       We split into tiles, keep only tiles whose histogram is actually
       bimodal, and derive the threshold from those. (Chini et al. 2017;
       Martinis et al. 2009 use the same idea for automatic near-real-time
       flood mapping.)

    2. TERRAIN MASKING. Radar shadow on slopes is also dark and is the classic
       false positive. Water does not sit on a hillside, so we reject
       candidate pixels whose HAND is implausibly high.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "to_db",
    "lee_filter",
    "otsu_threshold",
    "detect_water_otsu",
    "OtsuResult",
]


@dataclass
class OtsuResult:
    """Water mask plus everything needed to explain it."""

    mask: np.ndarray            # uint8 {0,1}
    confidence: np.ndarray      # float32 0..1
    threshold_db: float
    tiles_used: int
    tiles_total: int
    method: str = "otsu-tiled"
    # True where the sensor actually observed. Outside a satellite swath we
    # know NOTHING, and "not observed" must never be rendered as "dry".
    observed: Optional[np.ndarray] = None
    calibrated: bool = True

    @property
    def coverage(self) -> float:
        return 1.0 if self.observed is None else float(self.observed.mean())


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def to_db(linear: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Linear backscatter (sigma0) -> decibels.

    Thresholding is done in dB because backscatter is log-normally
    distributed; in dB the land and water modes are roughly Gaussian, which is
    the assumption Otsu's between-class variance criterion is built on.
    """
    arr = np.asarray(linear, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(arr, floor))


def lee_filter(image: np.ndarray, size: int = 5, cu: float = 0.523) -> np.ndarray:
    """Lee speckle filter (Lee, 1980).

    SAR speckle is multiplicative noise; unfiltered, it shatters the water
    mask into salt-and-pepper. The Lee filter smooths homogeneous areas while
    preserving edges — which is exactly what a flood boundary is.

    ``cu`` is the noise variation coefficient; 0.523 corresponds to
    single-look intensity data and is the usual default.
    """
    from scipy import ndimage

    img = np.asarray(image, dtype=np.float64)
    nan_mask = ~np.isfinite(img)
    if nan_mask.any():
        img = np.where(nan_mask, np.nanmean(img), img)

    mean = ndimage.uniform_filter(img, size)
    mean_sq = ndimage.uniform_filter(img ** 2, size)
    var = np.maximum(mean_sq - mean ** 2, 0.0)

    # Weight -> 1 at edges (keep the pixel), -> 0 in flat areas (use the mean).
    ci2 = np.divide(var, np.maximum(mean ** 2, 1e-12))
    weight = np.clip(1.0 - (cu ** 2) / np.maximum(ci2, 1e-12), 0.0, 1.0)

    out = mean + weight * (img - mean)
    if nan_mask.any():
        out[nan_mask] = np.nan
    return out


# --------------------------------------------------------------------------
# Otsu
# --------------------------------------------------------------------------
def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: the split maximising between-class variance.

    Implemented directly rather than imported so the baseline has no
    scikit-image dependency — the fallback path must always be available.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("otsu_threshold received no finite values")
    if float(np.ptp(v)) == 0.0:
        return float(v[0])

    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(np.median(v))

    w0 = np.cumsum(hist) / total                       # weight of class below
    w1 = 1.0 - w0                                      # weight of class above
    cum_mean = np.cumsum(hist * centers) / total
    global_mean = cum_mean[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, cum_mean / w0, 0.0)
        mu1 = np.where(w1 > 0, (global_mean - cum_mean) / w1, 0.0)
        between = w0 * w1 * (mu0 - mu1) ** 2

    between[~np.isfinite(between)] = 0.0
    return float(centers[int(np.argmax(between))])


def _bimodality_score(tile: np.ndarray) -> float:
    """How bimodal is this tile? Higher is better.

    A tile split by Otsu into two well-separated, reasonably-balanced classes
    is one where the threshold is meaningful. A tile of pure land gives two
    arbitrary halves of one blob and must be discarded.

    Score = (class separation, in units of pooled spread) x (balance penalty).
    """
    v = tile[np.isfinite(tile)]
    if v.size < 64:
        return 0.0
    try:
        t = otsu_threshold(v)
    except ValueError:
        return 0.0
    lo, hi = v[v <= t], v[v > t]
    if lo.size < 16 or hi.size < 16:
        return 0.0

    separation = float(hi.mean() - lo.mean())
    spread = float(np.sqrt(lo.var() + hi.var())) + 1e-9
    balance = min(lo.size, hi.size) / v.size          # 0.5 is perfectly split
    return (separation / spread) * balance


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def detect_water_otsu(
    vv: np.ndarray,
    vh: Optional[np.ndarray] = None,
    hand: Optional[np.ndarray] = None,
    *,
    already_db: bool = False,
    tile_size: int = 128,
    min_tiles: int = 4,
    tile_quantile: float = 0.85,
    speckle_filter: bool = True,
    hand_limit_m: float = 15.0,
    min_blob_pixels: int = 20,
    fallback_threshold_db: float = -16.0,
    max_water_threshold_db: float = -13.0,
    calibrated: bool = True,
) -> OtsuResult:
    """Detect open water in a Sentinel-1 scene.

    Parameters
    ----------
    vv, vh
        Backscatter arrays. VV is the primary channel for open water; VH is
        used only to sharpen confidence, never to override VV.
    hand
        Height Above Nearest Drainage. Used to veto radar shadow on terrain.
    already_db
        Set when the input is already in decibels.
    hand_limit_m
        Reject candidate water more than this far above the nearest channel.
        Generous on purpose — an extreme Brahmaputra flood really does inundate
        several metres above the channel, and over-tight terrain masking would
        erase exactly the event we care about.
    fallback_threshold_db
        Used only when no tile is bimodal, i.e. the scene contains no
        detectable water. Around -16 dB is the conventional open-water cut for
        C-band VV.
    max_water_threshold_db
        PHYSICAL CEILING on what may be called water, and it is load-bearing.

        Tiled Otsu has a nasty failure mode when a scene contains little
        water: a tile of pure LAND is often bimodal for a completely different
        reason — dark bare fields against radar-bright built-up area, which
        double-bounces off walls. Otsu happily splits that pair, the tile
        scores as "bimodal", and the derived threshold lands somewhere around
        -10 dB, which classifies every dry field in the district as flood.
        Left unchecked this reports MORE inundation at low water than at high,
        which is exactly backwards.

        Open water at C-band VV sits below roughly -15 dB. Refusing any
        threshold above -13 dB rejects those land-on-land splits while leaving
        genuine water detection untouched. This is a physical constraint, not
        a tuned fudge factor.
    calibrated
        Whether the input is real sigma0/gamma0 backscatter. Set False for raw
        Sentinel-1 GRD digital numbers, where an absolute dB value carries no
        physical meaning. When False, the ceiling and the fixed fallback are
        derived from the scene's own histogram instead of from published
        thresholds — applying a -13 dB cut to data that runs +15..+27 would
        classify the entire district as either all water or all land.

    Returns
    -------
    OtsuResult
        ``confidence`` is a smooth 0..1 field derived from distance below the
        threshold, not a constant. Downstream, low-confidence water is shown
        hatched rather than solid — an honest map is a better map.
    """
    vv_db = np.asarray(vv, dtype=np.float64) if already_db else to_db(vv)
    if speckle_filter:
        vv_db = lee_filter(vv_db, size=5)

    finite_vv = vv_db[np.isfinite(vv_db)]
    if finite_vv.size == 0:
        raise ValueError("SAR scene contains no finite pixels over this grid")

    if not calibrated:
        # No physical scale, so both limits come from the scene itself.
        #
        # The ceiling must express the same IDEA as the calibrated -13 dB cut —
        # "the split has to sit in the dark part of the histogram" — without
        # assuming any absolute value. The 60th percentile does that: land
        # normally dominates a scene, so a genuine land/water split falls below
        # it, while a land/built-up split (both bright) falls above it and is
        # rejected.
        #
        # A tighter cut such as the 20th percentile FAILS whenever flooding is
        # substantial, because the percentile then lands inside the water mode
        # itself and vetoes the correct threshold — under-detecting exactly
        # when the flood is worst.
        max_water_threshold_db = float(np.percentile(finite_vv, 60))
        fallback_threshold_db = float(np.percentile(finite_vv, 12))
        logger.warning(
            "Uncalibrated SAR input: physical backscatter thresholds do not apply. "
            "Deriving ceiling %.2f and fallback %.2f from this scene's own histogram. "
            "Prefer a calibrated RTC product for anything quotable.",
            max_water_threshold_db, fallback_threshold_db,
        )

    h, w = vv_db.shape
    scores: list[tuple[float, float]] = []   # (score, threshold)
    tiles_total = 0
    rejected_bright = 0

    for r0 in range(0, h, tile_size):
        for c0 in range(0, w, tile_size):
            tile = vv_db[r0 : r0 + tile_size, c0 : c0 + tile_size]
            if tile.size < 256:
                continue
            tiles_total += 1
            score = _bimodality_score(tile)
            if score <= 0:
                continue
            try:
                tile_threshold = otsu_threshold(tile)
            except ValueError:
                continue
            # Physical veto: a split above the open-water ceiling is a
            # land/built-up split, not a land/water one. See the docstring.
            if tile_threshold > max_water_threshold_db:
                rejected_bright += 1
                continue
            scores.append((score, tile_threshold))

    if rejected_bright:
        logger.debug(
            "%d tile(s) were bimodal but split above %.1f dB — land/built-up contrast, "
            "not water. Rejected.", rejected_bright, max_water_threshold_db,
        )

    if scores:
        scores.sort(key=lambda s: -s[0])
        cutoff = max(min_tiles, int(len(scores) * (1.0 - tile_quantile)))
        selected = scores[:cutoff]
        # Median, not mean: one pathological tile should not move the threshold.
        threshold = float(np.median([t for _s, t in selected]))
        tiles_used = len(selected)
        method = "otsu-tiled"
    else:
        logger.info(
            "No tile produced a valid water split in this scene. That normally means "
            "there is little or no open water. Falling back to the conventional C-band "
            "VV open-water cut of %.1f dB.",
            fallback_threshold_db,
        )
        threshold = fallback_threshold_db
        tiles_used = 0
        method = "otsu-fallback-fixed"

    # Belt and braces: even a median over accepted tiles must respect the
    # physical ceiling.
    threshold = min(threshold, max_water_threshold_db)

    candidate = (vv_db < threshold) & np.isfinite(vv_db)

    # --- terrain veto ------------------------------------------------------
    if hand is not None:
        terrain_ok = ~np.isfinite(hand) | (np.asarray(hand) <= hand_limit_m)
        rejected = int((candidate & ~terrain_ok).sum())
        if rejected:
            logger.info("Terrain veto removed %d px of implausibly high 'water'", rejected)
        candidate &= terrain_ok

    # --- morphological cleanup --------------------------------------------
    mask = _remove_small_blobs(candidate, min_blob_pixels)

    # --- confidence --------------------------------------------------------
    # How far below the threshold a pixel sits, normalised over ~4 dB. A pixel
    # 4 dB darker than the cut is unambiguous water; one 0.1 dB below is not.
    margin = (threshold - vv_db) / 4.0
    confidence = np.clip(margin, 0.0, 1.0)
    if vh is not None:
        vh_db = np.asarray(vh, dtype=np.float64) if already_db else to_db(vh)
        # VH is dominated by volume scattering, so it is even darker over open
        # water. Agreement between channels raises confidence; it never
        # creates water on its own.
        vh_support = np.clip((-22.0 - vh_db) / 4.0, 0.0, 1.0)
        confidence = 0.7 * confidence + 0.3 * vh_support
    confidence = np.where(mask > 0, np.clip(confidence, 0.05, 1.0), 0.0)

    observed = np.isfinite(vv_db)
    coverage = float(observed.mean())
    logger.info(
        "Otsu baseline: threshold %.2f dB from %d/%d tiles -> %.2f%% water "
        "(%.0f%% of the grid observed)",
        threshold, tiles_used, tiles_total, 100.0 * mask.mean(), 100.0 * coverage,
    )
    if coverage < 0.999:
        logger.warning(
            "%.0f%% of the grid was NOT OBSERVED by this scene. Flood extent there is "
            "UNKNOWN, not zero — treat the reported extent as a lower bound.",
            100.0 * (1.0 - coverage),
        )

    return OtsuResult(
        mask=mask.astype(np.uint8),
        confidence=confidence.astype(np.float32),
        threshold_db=threshold,
        tiles_used=tiles_used,
        tiles_total=tiles_total,
        method=method,
        observed=observed,
        calibrated=calibrated,
    )


def _remove_small_blobs(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop speckle-sized connected components.

    At 30 m, 20 pixels is ~1.8 ha. Below that it is almost certainly residual
    speckle or a wet field, not a flood a responder needs routing around.
    """
    if min_pixels <= 1:
        return mask
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = sizes >= min_pixels
    return keep[labels]
