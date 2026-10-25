"""PRAHARI — Flood Evacuation Decision Layer.

India can already predict floods. It still cannot tell a family which road is
safe. PRAHARI is the decision layer that turns a flood warning into a route.

The package is deliberately layered so that each block can be tested, and
defended to a panel, on its own:

    prahari.grid      spatial frame — metric raster grid, no heavy deps
    prahari.ingest    data spine — DEM, OSM, population, SAR, rainfall
    prahari.hazard    flood extent (Otsu baseline + U-Net) and depth (HAND)
    prahari.impact    flood x roads / people / shelters
    prahari.decision  THE DIFFERENTIATOR — prune, assign, route, prioritise
    prahari.delivery  responder dashboard payloads, action cards, CAP 1.2
    prahari.demo      synthetic + replay harnesses so the demo never dies
    prahari.validate  retrospective validation against real flood events

Import cost is kept low on purpose: importing `prahari` pulls in nothing
heavier than the standard library.
"""

import os as _os

# Anaconda ships its own OpenMP runtime (libiomp5md.dll) and so does PyTorch.
# On Windows, importing both aborts the process with "OMP: Error #15" before a
# single tensor is allocated — which on a stock Anaconda install is every
# machine most teams have.
#
# This lives at package level rather than next to the torch import because
# torch is reached from several modules (unet, infer, train) and whichever one
# gets there first must have already set it. Setting one environment variable
# costs nothing and keeps the package import cheap.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
__all__ = ["__version__"]
