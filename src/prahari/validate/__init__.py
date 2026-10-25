"""Retrospective validation — the highest-value thing this project can do.

Almost no student team validates. It takes wall-clock time rather than
cleverness, which is exactly why it is worth doing and exactly why it
differentiates. The target sentence is:

    "Run against the July 2024 event, PRAHARI flagged N villages as cut off.
     The actual relief operation reached those villages on day three."

What is honestly checkable without a ground-truth licence:

    permanent_water.py   separates FLOOD from RIVER. Detected water includes
                         the Brahmaputra itself; reporting the permanent
                         channel as flood inflates every number. JRC Global
                         Surface Water gives the permanent extent, and
                         subtracting it is a correctness fix as much as a
                         validation step.

    retrospective.py     runs the pipeline across a real event's Sentinel-1
                         series and checks the flood behaves like a flood:
                         rises, peaks, recedes. A detector that reports random
                         extents over a hydrograph is broken, and this catches
                         it without any external label.

    Optional: supply an independent reference mask (a Bhuvan/NDEM export, or a
    hand-digitised extent) with --reference and we score IoU/F1 against it.

We do NOT claim validation we have not done. Where a number is unverified, the
report says so.
"""

from .permanent_water import fetch_permanent_water, separate_flood_from_permanent
from .retrospective import run_retrospective, RetrospectiveReport

__all__ = [
    "fetch_permanent_water",
    "separate_flood_from_permanent",
    "run_retrospective",
    "RetrospectiveReport",
]
