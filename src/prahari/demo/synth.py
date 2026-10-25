"""Synthetic district generator — a plausible Brahmaputra floodplain.

Purpose: exercise the ENTIRE pipeline, including SAR water detection, with no
downloads, no API keys and no network. It exists for three reasons:

    1. development and CI, before the real data lands;
    2. the "no internet at the venue" contingency;
    3. so every component is testable in isolation with known ground truth.

It generates terrain that behaves like the real thing in the ways that matter
to this system:

  * a braided, meandering main channel, because Brahmaputra anabranches are
    what make bridge crossings scarce and settlements strandable;
  * an almost flat floodplain, which is the condition that breaks naive D8 and
    forces the flat-resolution step in hand.py to earn its place;
  * road networks that cross the river at only a FEW bridges — the structural
    reason a district fragments into disconnected components when water rises;
  * villages sited near the river, because that is where the land is fertile
    and where people actually live.

Then it renders a synthetic Sentinel-1 scene from the flood state, complete
with speckle, so the Otsu baseline and the U-Net have something realistic to
segment rather than being handed a clean mask.

EVERYTHING produced here is stamped ``is_synthetic=True`` and propagates that
flag to the dashboard banner and the API response.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..config import DistrictConfig
from ..grid import Grid, grid_from_config
from ..scenario import Scenario, Settlement, Shelter

logger = logging.getLogger(__name__)

__all__ = ["SynthConfig", "synthesise_district", "synthesise_sar"]


@dataclass
class SynthConfig:
    seed: int = 20260815
    n_villages: int = 64
    # CHAR villages: settlements on the sandbar islands and immediate banks of
    # the Brahmaputra. Barpeta has a great many of them, they hold real
    # populations, and they are the settlements that strand EVERY year — they
    # are the reason a boat dispatch list is the operationally valuable
    # output. Omitting them would make the synthetic district far too tidy and
    # would quietly delete the case the whole system exists to handle.
    n_char_villages: int = 14
    # Barpeta district has hundreds of schools; every village of any size has
    # one, and schools are what Assam actually uses as relief camps.
    n_shelters: int = 120
    population_total: float = 1_690_000.0     # Barpeta's order of magnitude
    n_bridges: int = 4
    road_segment_m: float = 250.0
    channel_depth_m: float = 10.0
    floodplain_slope: float = 0.00035
    # Correlated micro-relief amplitude. Kept small on purpose: large random
    # relief carves spurious hollows across the plain, every hollow becomes a
    # "channel" once flow accumulates, and HAND then collapses toward zero
    # everywhere — which makes a 2 m rise look like the end of the world.
    micro_relief_m: float = 0.35
    downsample: int = 1                        # >1 shrinks the grid for speed


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------
def _build_terrain(grid: Grid, cfg: SynthConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dem, channel_distance)``.

    The channel is a sum of sinusoids rather than a single sine so that the
    meander does not repeat with an obvious period — a periodic river makes
    the whole demo look generated.
    """
    h, w = grid.shape
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)

    centre = h * 0.52
    for wavelength, amplitude, phase in (
        (w / 1.7, h * 0.16, rng.uniform(0, 2 * np.pi)),
        (w / 4.3, h * 0.07, rng.uniform(0, 2 * np.pi)),
        (w / 9.1, h * 0.025, rng.uniform(0, 2 * np.pi)),
    ):
        centre = centre + amplitude * np.sin(2 * np.pi * cols / wavelength + phase)

    channel_distance = np.abs(rows - centre) * grid.resolution

    # Regional slope: the Brahmaputra floodplain falls gently west-to-east.
    dem = 32.0 + cfg.floodplain_slope * cols * grid.resolution

    # Incise the main channel, plus a narrower anabranch offset from it.
    main_width = 420.0
    dem -= cfg.channel_depth_m * np.exp(-(channel_distance ** 2) / (2 * main_width ** 2))
    anabranch = np.abs(rows - (centre + h * 0.085)) * grid.resolution
    dem -= (cfg.channel_depth_m * 0.45) * np.exp(-(anabranch ** 2) / (2 * 180.0 ** 2))

    # Natural levees: the ground immediately beside a channel is HIGHER than
    # the floodplain behind it. This is why breaches matter, and why flooding
    # is not simply "closest to the river gets wet first".
    levee = 1.4 * np.exp(-((channel_distance - 520.0) ** 2) / (2 * 260.0 ** 2))
    dem += levee

    # Backswamp depressions behind the levees.
    for _ in range(9):
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        ry = rng.uniform(h * 0.03, h * 0.10)
        rx = rng.uniform(w * 0.03, w * 0.12)
        blob = np.exp(-(((rows - cy) / ry) ** 2 + ((cols - cx) / rx) ** 2))
        dem -= rng.uniform(0.5, 1.6) * blob

    # Correlated micro-relief. Uncorrelated noise would be filled away
    # instantly by the depression fill and teach us nothing.
    from scipy import ndimage

    noise = ndimage.gaussian_filter(rng.normal(0, 1.0, (h, w)), sigma=6.0)
    spread = float(np.ptp(noise))
    dem += cfg.micro_relief_m * noise / (spread if spread > 0 else 1.0)

    return dem.astype(np.float32), channel_distance.astype(np.float32)


# --------------------------------------------------------------------------
# Settlements and population
# --------------------------------------------------------------------------
def _place_villages(
    grid: Grid, dem: np.ndarray, channel_distance: np.ndarray, cfg: SynthConfig,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    """Return ``[(x, y, population, is_char)]``.

    Two settlement populations, because Barpeta genuinely has two:

      MAINLAND villages sit 300 m to 4 km from the channel — close enough for
      water and fertile silt, far enough to be off the active bed. This is
      where most people live, and it is already dangerous because it is on the
      floodplain.

      CHAR villages sit on the sandbars and immediate banks, 120-700 m out.
      They are low, they flood first, and they have at most one or two road
      links to the mainland. When those go, the village is an island. That is
      not a synthetic quirk — it is the defining feature of flooding in this
      district and the reason the boat dispatch list matters.
    """
    h, w = grid.shape
    mainland: list[tuple[float, float, float, bool]] = []
    chars: list[tuple[float, float, float, bool]] = []

    n_main = cfg.n_villages
    n_char = cfg.n_char_villages

    # Heavy-tailed size distribution: a few towns, many hamlets.
    weights = rng.pareto(1.6, n_main) + 1.0
    weights = weights / weights.sum() * (cfg.population_total * 0.93)
    # Char populations are real but individually smaller.
    char_weights = rng.uniform(0.6, 1.6, max(n_char, 1))
    char_weights = char_weights / char_weights.sum() * (cfg.population_total * 0.07)

    def _try_place(target: list, count: int, lo: float, hi: float, spacing: float,
                   sizes: np.ndarray, is_char: bool) -> None:
        attempts = 0
        while len(target) < count and attempts < count * 600:
            attempts += 1
            r = int(rng.integers(int(h * 0.03), int(h * 0.97)))
            c = int(rng.integers(int(w * 0.03), int(w * 0.97)))
            dist = float(channel_distance[r, c])
            if not (lo < dist < hi):
                continue
            x, y = grid.xy(r, c)
            occupied = mainland + chars
            if any(math.hypot(float(x) - px, float(y) - py) < spacing for px, py, _, _ in occupied):
                continue
            target.append((float(x), float(y), float(sizes[len(target)]), is_char))

    _try_place(mainland, n_main, 300.0, 4200.0, 1500.0, weights, False)
    # Lower bound of 260 m keeps the char settlement itself outside the
    # channel buffer used by the road builder, so it can hold a road node at
    # all — an island with no road stub is unroutable rather than fragile, and
    # fragile is the behaviour we are modelling.
    _try_place(chars, n_char, 260.0, 700.0, 900.0, char_weights, True)

    logger.info(
        "Synth: placed %d mainland villages and %d char (river-island) villages",
        len(mainland), len(chars),
    )
    return mainland + chars


def _population_raster(
    grid: Grid, villages: list[tuple[float, float, float]], rng: np.random.Generator
) -> np.ndarray:
    """Gaussian population blobs plus a thin rural background."""
    h, w = grid.shape
    pop = np.zeros((h, w), dtype=np.float64)
    X, Y = grid.coordinate_arrays()

    largest = max(p for _, _, p, _ in villages)
    for x, y, people, is_char in villages:
        # Char settlements are compact — there is only so much sandbar.
        sigma = (150.0 if is_char else 260.0) + 460.0 * (people / largest) ** 0.5
        blob = np.exp(-(((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2)))
        total = blob.sum()
        if total > 0:
            pop += blob * (people * 0.86 / total)

    # 14% of people live scattered outside village cores.
    background = rng.random((h, w)) ** 3
    scattered = sum(p for _, _, p, _ in villages) * 0.14
    pop += background * (scattered / max(background.sum(), 1.0))
    return pop.astype(np.float32)


# --------------------------------------------------------------------------
# Road network
# --------------------------------------------------------------------------
def _build_roads(
    grid: Grid,
    villages: list[tuple[float, float, float]],
    dem: np.ndarray,
    channel_distance: np.ndarray,
    cfg: SynthConfig,
    rng: np.random.Generator,
):
    """Build a routable road graph connecting the villages.

    Delaunay triangulation gives a realistic-looking network: neighbouring
    settlements are linked, distant ones are not, and there are alternative
    paths without the network being complete.

    The critical structural detail is the RIVER. Links that would cross the
    main channel are dropped unless they pass close to one of a small number
    of bridge points. Four bridges across a braided Brahmaputra channel is
    generous by real standards, and it is what makes losing one of them
    fragment the district — which is the behaviour the whole priority list
    depends on.
    """
    import networkx as nx
    from scipy.spatial import Delaunay

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = grid.crs

    points = np.array([[x, y] for x, y, _, _ in villages])
    is_char = [bool(v[3]) for v in villages]
    if len(points) < 4:
        raise ValueError("need at least 4 villages to triangulate a road network")

    # Bridge points along the channel centreline.
    h, w = grid.shape
    channel_cells = np.argwhere(channel_distance < grid.resolution * 1.5)
    if channel_cells.size == 0:
        channel_cells = np.argwhere(channel_distance < channel_distance.min() + grid.resolution * 3)
    picks = np.linspace(0, len(channel_cells) - 1, cfg.n_bridges + 2)[1:-1].astype(int)
    bridges = []
    for i in picks:
        r, c = channel_cells[i]
        bx, by = grid.xy(int(r), int(c))
        bridges.append((float(bx), float(by)))
    logger.info("Synth: %d bridges across the main channel", len(bridges))

    tri = Delaunay(points)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(simplex[a]), int(simplex[b])
            edges.add((min(i, j), max(i, j)))

    # Char villages get at most two road links, because that is what a sandbar
    # settlement actually has: an embankment track and maybe a second one.
    # Delaunay would otherwise hand them five or six neighbours and make them
    # as well connected as a town, which erases the stranding behaviour.
    edges = _thin_char_links(edges, points, is_char, max_links=2)

    node_id = 0
    village_nodes: dict[int, int] = {}
    for idx, (x, y, _, char_flag) in enumerate(villages):
        graph.add_node(node_id, x=float(x), y=float(y), village=idx, char=bool(char_flag))
        village_nodes[idx] = node_id
        node_id += 1

    kept = dropped = 0
    for i, j in sorted(edges):
        x0, y0, _, _ = villages[i]
        x1, y1, _, _ = villages[j]
        waypoints = _river_aware_path((x0, y0), (x1, y1), grid, channel_distance, bridges)
        if waypoints is None:
            dropped += 1
            continue
        kept += 1

        # Classify: long links become the arterial network.
        span = math.hypot(x1 - x0, y1 - y0)
        highway = "primary" if span > 9000 else ("secondary" if span > 5000 else "residential")

        prev = village_nodes[i]
        for k, (wx, wy) in enumerate(waypoints[1:-1], start=1):
            graph.add_node(node_id, x=float(wx), y=float(wy))
            _add_road(graph, prev, node_id, highway)
            prev = node_id
            node_id += 1
        _add_road(graph, prev, village_nodes[j], highway)

    logger.info(
        "Synth: road network %d nodes / %d directed edges (%d village links kept, %d dropped "
        "as uncrossable river)",
        graph.number_of_nodes(), graph.number_of_edges(), kept, dropped,
    )
    return graph, bridges


def _thin_char_links(
    edges: set[tuple[int, int]],
    points: np.ndarray,
    is_char: list[bool],
    max_links: int = 2,
) -> set[tuple[int, int]]:
    """Keep only the ``max_links`` shortest links for each char village."""
    by_char: dict[int, list[tuple[float, tuple[int, int]]]] = {}
    other: set[tuple[int, int]] = set()

    for i, j in edges:
        char_ends = [n for n in (i, j) if is_char[n]]
        if not char_ends:
            other.add((i, j))
            continue
        span = float(np.hypot(*(points[i] - points[j])))
        for n in char_ends:
            by_char.setdefault(n, []).append((span, (i, j)))

    kept: set[tuple[int, int]] = set(other)
    for _n, options in by_char.items():
        options.sort()
        for _span, edge in options[:max_links]:
            kept.add(edge)
    return kept


def _river_aware_path(
    start: tuple[float, float],
    end: tuple[float, float],
    grid: Grid,
    channel_distance: np.ndarray,
    bridges: list[tuple[float, float]],
    segment_m: float = 250.0,
) -> Optional[list[tuple[float, float]]]:
    """Densified polyline from ``start`` to ``end``, or None if it must cross
    the channel away from a bridge."""
    def crosses_channel(a: tuple[float, float], b: tuple[float, float]) -> bool:
        n = max(int(math.dist(a, b) / (grid.resolution * 0.8)), 2)
        for t in np.linspace(0, 1, n):
            px = a[0] + t * (b[0] - a[0])
            py = a[1] + t * (b[1] - a[1])
            d = grid.sample(channel_distance, px, py, default=1e9)
            if float(d) < 200.0:
                return True
        return False

    if not crosses_channel(start, end):
        return _densify([start, end], segment_m)

    # Must cross: detour via the nearest bridge that makes both legs clean.
    best: Optional[list[tuple[float, float]]] = None
    best_len = math.inf
    for bx, by in bridges:
        bridge = (bx, by)
        detour = math.dist(start, bridge) + math.dist(bridge, end)
        if detour > 3.2 * max(math.dist(start, end), 1.0) or detour >= best_len:
            continue
        best = _densify([start, bridge, end], segment_m)
        best_len = detour
    return best


def _densify(points: list[tuple[float, float]], segment_m: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        span = math.dist(a, b)
        n = max(int(span / segment_m), 1)
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


_ROAD_NAMES = [
    "Barpeta Road", "Jania Road", "Sarthebari Link", "Howly Main Road", "Pathsala Road",
    "Kalgachia Road", "Mandia Road", "Chenga Road", "Bhabanipur Road", "Sarupeta Road",
    "Baghbar Road", "Gobardhana Road", "Embankment Road", "Nagaon Link", "Sorbhog Road",
]


def _add_road(graph, u: int, v: int, highway: str) -> None:
    x0, y0 = graph.nodes[u]["x"], graph.nodes[u]["y"]
    x1, y1 = graph.nodes[v]["x"], graph.nodes[v]["y"]
    length = math.hypot(x1 - x0, y1 - y0)
    name = _ROAD_NAMES[(u * 31 + v * 17) % len(_ROAD_NAMES)]
    attrs = {"length": length, "highway": highway, "name": name}
    graph.add_edge(u, v, key=0, **attrs)
    graph.add_edge(v, u, key=0, **attrs)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
_SHELTER_KINDS = ["school", "school", "school", "college", "community_centre", "hospital"]

# Several shelters can sit in one village, so names must not collide — a
# duplicate name in a dispatch list is an operational hazard, not a cosmetic
# bug: two different buildings that read identically get conflated on the radio.
_SHELTER_LABELS = {
    "school": ["Higher Secondary School", "ME School", "LP School", "Girls' High School"],
    "college": ["College", "Junior College", "Polytechnic"],
    "community_centre": ["Community Centre", "Panchayat Bhawan", "Namghar Hall"],
    "hospital": ["Community Health Centre", "Primary Health Centre", "Sub-Divisional Hospital"],
}


def synthesise_district(
    config: DistrictConfig,
    synth: Optional[SynthConfig] = None,
) -> Scenario:
    """Generate a complete synthetic district: terrain, HAND, roads, people, shelters."""
    from ..hazard.hand import compute_hand

    synth = synth or SynthConfig()
    rng = np.random.default_rng(synth.seed)

    grid = grid_from_config(config)
    if synth.downsample > 1:
        grid = Grid(
            origin_x=grid.origin_x, origin_y=grid.origin_y,
            resolution=grid.resolution * synth.downsample,
            width=grid.width // synth.downsample,
            height=grid.height // synth.downsample,
            crs=grid.crs,
        )
    logger.info("Synth: generating %s on %s", config.name, grid)

    dem, channel_distance = _build_terrain(grid, synth, rng)

    hand_result = compute_hand(
        dem.astype(np.float64),
        cell_size=grid.resolution,
        drainage_threshold_cells=config.hydrology.threshold_cells(grid.resolution),
        permanent_water=channel_distance < 200.0,
    )

    villages = _place_villages(grid, dem, channel_distance, synth, rng)
    population = _population_raster(grid, villages, rng)
    roads, bridges = _build_roads(grid, villages, dem, channel_distance, synth, rng)

    settlements = [
        Settlement(
            id=f"v{i:03d}",
            name=(f"{_village_name(i)} Char" if is_char else _village_name(i)),
            x=x, y=y,
            population=float(pop),
        )
        for i, (x, y, pop, is_char) in enumerate(villages)
    ]

    # Shelters spread across MAINLAND villages, largest first, wrapping so big
    # settlements get several and small ones get one — a town has four schools,
    # a hamlet has one.
    #
    # Char villages get NO shelter. Siting a relief camp on a sandbar that goes
    # under every monsoon would be absurd, and in practice char residents are
    # evacuated to the mainland. This is the structural reason char settlements
    # dominate the boat dispatch list.
    mainland_idx = [i for i, v in enumerate(villages) if not v[3]]
    order = sorted(mainland_idx, key=lambda i: -villages[i][2])
    shelters: list[Shelter] = []
    for n in range(synth.n_shelters):
        i = order[n % len(order)]
        x, y, _pop, _char = villages[i]
        # Offset from the village centroid so shelter and settlement are not
        # the same point — otherwise travel time is trivially zero and the
        # assignment problem becomes uninteresting.
        angle = rng.uniform(0, 2 * math.pi)
        offset = rng.uniform(280.0, 1100.0)
        sx = x + offset * math.cos(angle)
        sy = y + offset * math.sin(angle)
        footprint = float(rng.uniform(700.0, 3400.0))
        from ..impact.shelters import estimate_capacity_from_footprint

        capacity = estimate_capacity_from_footprint(
            footprint, config.evacuation.sqm_per_person, storeys=float(rng.integers(1, 3))
        )
        kind = _SHELTER_KINDS[n % len(_SHELTER_KINDS)]
        round_trip = n // len(order)          # 0 the first time we visit a village
        shelters.append(
            Shelter(
                id=f"h{n:03d}",
                name=f"{_village_name(i)} {_SHELTER_LABELS[kind][round_trip % len(_SHELTER_LABELS[kind])]}",
                x=float(sx), y=float(sy),
                capacity=max(capacity, 60),
                capacity_source="footprint_estimate",
                kind=kind,
                footprint_sqm=footprint,
            )
        )

    scenario = Scenario(
        district_id=config.district.id,
        grid=grid,
        dem=dem,
        hand=hand_result.hand,
        population=population,
        roads=roads,
        shelters=shelters,
        settlements=settlements,
        timestamp=datetime.now(timezone.utc).isoformat(),
        meta={
            "synthetic": True,
            "seed": synth.seed,
            "bridges": [[float(a), float(b)] for a, b in bridges],
            "drainage_cells": int(hand_result.drainage.sum()),
        },
    )
    for layer, detail in (
        ("dem", "procedural Brahmaputra-style floodplain"),
        ("hand", "computed from the synthetic DEM by the real HAND pipeline"),
        ("population", "procedural village clusters"),
        ("roads", "Delaunay network with a limited number of bridge crossings"),
        ("shelters", "capacity estimated from synthetic building footprints"),
    ):
        scenario.add_provenance(layer, "synthetic", detail=detail, is_synthetic=True, is_estimate=True)

    logger.info(
        "Synth: %s ready — %d settlements (%.0f people), %d shelters (%d places), %d road edges",
        config.name, len(settlements), scenario.total_population(),
        len(shelters), sum(s.capacity for s in shelters), roads.number_of_edges(),
    )
    return scenario


_VILLAGE_NAMES = [
    "Jania", "Howly", "Sarthebari", "Pathsala", "Kalgachia", "Mandia", "Chenga",
    "Bhabanipur", "Sarupeta", "Baghbar", "Gobardhana", "Sorbhog", "Barpeta Town",
    "Bahari", "Rupshi", "Moinbari", "Kaljar", "Bhella", "Dolbari", "Nagaon",
    "Simlaguri", "Pakabetbari", "Kayakuchi", "Barbala", "Bhawanipur", "Uttar Kalgachia",
    "Dakshin Bhabanipur", "Madhya Jania", "Bagribari", "Chowkbazar", "Salbari",
    "Kachumara", "Gomaphulbari", "Barpeta Ghat", "Nichuka", "Jhaubari", "Balikuri",
    "Khandakarpara", "Sundaridiya", "Bhakuamari", "Tarabari", "Alopati", "Mazergaon",
    "Sonkuchi", "Barnagar", "Athiabari", "Bilartari", "Dhaniram", "Kaldoba",
    "Manikpur", "Bharatpur", "Ghograpar", "Namkhola", "Bausimari", "Silarchar",
    "Purabhitha", "Doulasal", "Hazipara", "Kharbuja", "Modati", "Baniakuchi",
    "Jotigaon", "Sialmari", "Amguri", "Sonapur", "Dhepargaon", "Ramdiya",
    "Bakuljhora", "Rangapani", "Nowapara", "Kachaimari",
]


def _village_name(i: int) -> str:
    if i < len(_VILLAGE_NAMES):
        return _VILLAGE_NAMES[i]
    return f"Village {i + 1}"


# --------------------------------------------------------------------------
# Synthetic SAR
# --------------------------------------------------------------------------
def synthesise_sar(
    scenario: Scenario,
    stage_m: float,
    seed: int = 0,
    speckle_db: float = 1.7,
    permanent_water_stage_m: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a synthetic Sentinel-1 scene for a given river stage.

    Returns ``(vv_db, vh_db, truth_mask)``.

    ``stage_m`` is the water level above the channel, expressed on the HAND
    datum: every cell with ``HAND <= stage_m`` is inundated. That is the same
    physical assumption the depth model makes, which is exactly why the truth
    mask returned here is a fair test of the DETECTOR but NOT of the depth
    model — the depth model would be marking its own homework. The validation
    harness tests depth against real events instead.

    Speckle is added because a clean mask makes the Otsu baseline look far
    better than it is in reality, and a demo tuned on noiseless data falls
    over on the first real scene.
    """
    rng = np.random.default_rng(seed)
    hand = np.nan_to_num(np.asarray(scenario.hand, dtype=np.float64), nan=1e6)

    truth = (hand <= max(stage_m, permanent_water_stage_m)).astype(np.uint8)

    # Backscatter means, in dB. Open water ~-20, vegetated land ~-8.
    vv = np.where(truth > 0, -20.0, -8.2)
    vh = np.where(truth > 0, -27.0, -15.4)

    # Land is not uniform: fields, built-up, vegetation all differ.
    from scipy import ndimage

    texture = ndimage.gaussian_filter(rng.normal(0, 1, hand.shape), sigma=4.0)
    spread = float(np.std(texture)) or 1.0
    vv = vv + np.where(truth > 0, 0.4, 2.1) * texture / spread
    vh = vh + np.where(truth > 0, 0.5, 2.4) * texture / spread

    # Built-up areas are radar-BRIGHT (double-bounce off walls) even when
    # flooded, which is a genuine limitation of SAR flood mapping: urban
    # flooding is systematically under-detected. Modelling it here keeps the
    # demo honest about a real weakness.
    built = np.asarray(scenario.population, dtype=np.float64)
    if built.max() > 0:
        built_norm = np.clip(built / np.percentile(built[built > 0], 97), 0, 1)
        vv = vv + 6.5 * built_norm
        vh = vh + 5.0 * built_norm

    vv = vv + rng.normal(0, speckle_db, hand.shape)
    vh = vh + rng.normal(0, speckle_db * 1.1, hand.shape)

    return vv.astype(np.float32), vh.astype(np.float32), truth
