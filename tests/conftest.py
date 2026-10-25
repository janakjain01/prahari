"""Shared fixtures.

Everything here is small and deterministic. The test suite must run offline,
in seconds, with no downloads — a suite that needs the network is a suite
nobody runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prahari.config import load_config          # noqa: E402
from prahari.grid import Grid                   # noqa: E402
from prahari.scenario import Scenario, Settlement, Shelter  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config("barpeta")


@pytest.fixture
def small_grid() -> Grid:
    """A 60x80 grid at 30 m — big enough to be interesting, small enough to be instant."""
    return Grid(origin_x=300_000.0, origin_y=2_900_000.0, resolution=30.0,
                width=80, height=60, crs="EPSG:32646")


@pytest.fixture
def valley_dem(small_grid) -> np.ndarray:
    """A river valley running west-east with a gentle downstream slope."""
    h, w = small_grid.shape
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
    centre = h * 0.5 + 6.0 * np.sin(cols / 11.0)
    distance = np.abs(rows - centre)
    dem = 40.0 + 0.02 * cols
    dem += np.clip(distance, 0, None) * 0.35
    dem -= 7.0 * np.exp(-(distance ** 2) / (2 * 3.0 ** 2))
    return dem


@pytest.fixture
def toy_graph():
    """A deliberately simple network with a known correct answer.

        v0 --- j1 --- h0        (short, will be flooded in tests)
         |             |
        v1 ---------- j2 --- h1 (long detour, stays dry)

    Small enough that every routing and assignment assertion can be verified
    by hand on paper — which is the point. A test whose expected value came
    out of the code it is testing proves nothing.
    """
    import networkx as nx

    g = nx.MultiDiGraph()
    coords = {
        "v0": (0.0, 0.0), "v1": (0.0, -600.0),
        "j1": (500.0, 0.0), "j2": (900.0, -600.0),
        "h0": (1000.0, 0.0), "h1": (1500.0, -600.0),
    }
    for node, (x, y) in coords.items():
        g.add_node(node, x=x, y=y)

    def link(a, b, name):
        import math

        (x0, y0), (x1, y1) = coords[a], coords[b]
        length = math.hypot(x1 - x0, y1 - y0)
        for u, v in ((a, b), (b, a)):
            g.add_edge(u, v, key=0, length=length, name=name, highway="residential")

    link("v0", "j1", "Short Road")
    link("j1", "h0", "Short Road")
    link("v0", "v1", "Village Link")
    link("v1", "j2", "Long Road")
    link("j2", "h1", "Long Road")
    link("j2", "h0", "Cross Road")
    return g


@pytest.fixture
def toy_scenario(small_grid, valley_dem, toy_graph, config) -> Scenario:
    """A complete, tiny scenario with hazard already applied."""
    from prahari.hazard.hand import compute_hand

    hand = compute_hand(valley_dem, cell_size=30.0, drainage_threshold_cells=40).hand

    population = np.zeros(small_grid.shape, dtype=np.float32)
    population[10:20, 5:15] = 12.0       # a village up near v0
    population[45:55, 5:15] = 8.0        # another near v1

    flood = (hand < 1.2).astype(np.uint8)
    depth = np.where(flood > 0, 0.45, 0.0).astype(np.float32)

    settlements = [
        Settlement(id="v0", name="Upper Village", x=0.0, y=0.0, population=1200.0,
                   people_at_risk=1200.0),
        Settlement(id="v1", name="Lower Village", x=0.0, y=-600.0, population=800.0,
                   people_at_risk=800.0),
    ]
    shelters = [
        Shelter(id="h0", name="Near School", x=1000.0, y=0.0, capacity=500,
                capacity_source="footprint_estimate"),
        Shelter(id="h1", name="Far School", x=1500.0, y=-600.0, capacity=2000,
                capacity_source="footprint_estimate"),
    ]

    return Scenario(
        district_id="test", grid=small_grid, dem=valley_dem.astype(np.float32),
        hand=hand, population=population, flood_mask=flood, depth=depth,
        confidence=np.where(flood > 0, 0.9, 0.0).astype(np.float32),
        roads=toy_graph, shelters=shelters, settlements=settlements,
    )
