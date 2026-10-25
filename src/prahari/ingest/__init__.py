"""Data spine — real observations, every one of them optional.

The architectural rule here is the one that decides whether this project
survives contact with reality:

    NO SINGLE FEED IS ALLOWED TO BE LOAD-BEARING.

Two of the feeds we want are gated behind APPROVAL QUEUES, not effort:

    IMD API              registration + IP whitelisting
    Google Flood Hub     API key + waitlist

Neither can be hurried. Apply for both on day one, then build as though
neither will ever arrive — because one of them probably will not, and a system
that stops working when a key does not show up is a system that fails for
reasons entirely outside the team's control.

What needs no waitlist and therefore forms the actual backbone:

    Sentinel-1 SAR       Microsoft Planetary Computer, free, STAC, no key
    Copernicus DEM       Planetary Computer
    OpenStreetMap        Overpass / osmnx
    WorldPop             direct download
    Open-Meteo           keyless precipitation, works today

Every module here exposes the same contract: try the real source, and on any
failure return a result carrying ``ok=False`` and a human-readable reason
rather than raising. The caller decides whether to degrade. Nothing crashes
the pipeline because a server was slow.
"""

from .base import IngestResult, requires
from .dem import fetch_dem
from .osm import (
    fetch_place_names,
    fetch_road_network,
    fetch_shelter_candidates,
    label_settlements,
)
from .sen1floods11 import download_sen1floods11
from .population import fetch_worldpop
from .sentinel1 import fetch_sentinel1_scene, search_sentinel1
from .weather import fetch_precipitation, fetch_river_forecast
from .build import build_real_scenario

__all__ = [
    "IngestResult",
    "requires",
    "fetch_dem",
    "fetch_road_network",
    "fetch_shelter_candidates",
    "fetch_place_names",
    "label_settlements",
    "download_sen1floods11",
    "fetch_worldpop",
    "fetch_sentinel1_scene",
    "search_sentinel1",
    "fetch_precipitation",
    "fetch_river_forecast",
    "build_real_scenario",
]
