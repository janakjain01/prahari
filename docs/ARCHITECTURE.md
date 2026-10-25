# Architecture

How PRAHARI works, and why each decision was made that way.

---

## The shape of the problem

The system's performance profile is lopsided, and the whole architecture follows it:

| | Cost | Changes during an event? |
|---|---|---|
| **Static** — terrain, HAND, road graph, population, shelters | Expensive (HAND dominates) | Never |
| **Dynamic** — flood extent, depth, and the entire decision chain | ~1 second | Every cycle |

Separating them is why the replay scrubber is instant: dragging the timeline recomputes
only the dynamic half over static layers already in memory. A demo that stalls twenty
seconds per frame is a demo nobody watches to the end.

---

## One grid, one CRS, one truth

Every raster in the system — DEM, HAND, flood mask, depth, population, confidence —
is a numpy array **on the same grid, in a projected CRS with metre units**.

That single decision removes an entire class of bugs:

- no resampling mismatches between layers;
- no degrees-vs-metres confusion in hydrology (doing flow routing in degrees is a
  correctness bug, not a style choice);
- spatial joins become array indexing rather than geometry operations.

`grid.py` owns the frame. Pixel convention is stated explicitly because getting it wrong
shifts every result by half a cell: `origin` is the **outer** top-left corner, rows
increase southward, and `xy(row, col)` returns the pixel **centre**.

The decision core depends on numpy alone. `pyproj` is imported lazily and only for
lon/lat conversion.

---

## Block 1 — Data spine

`ingest/` — every adapter obeys one contract:

```python
IngestResult(ok=False, reason="…")     # never raises for an expected failure
```

Network down, key missing, dependency absent, nothing in the archive — all return a
value. The reason is that ingestion runs in three contexts that want different responses:
the CLI wants to print and continue, the API wants to degrade to cache, the validation
harness wants to fail loudly. Raising takes that choice away from the caller.

**No feed is load-bearing.** The two we would most like — IMD and Google Flood Hub — are
behind *approval queues*, not tasks. No amount of effort makes them arrive faster, so
nothing depends on them.

Everything actually required is free and keyless: Sentinel-1 and Copernicus DEM via
Planetary Computer, OSM, WorldPop, JRC Global Surface Water, Open-Meteo.

---

## Block 2 — Hazard

### Two independent paths, on purpose

```
otsu.py    classical thresholding. No training, no GPU, no checkpoint. Always works.
unet.py    U-Net on Sen1Floods11. Better — but only ships if it BEATS the baseline.
```

The baseline is built first so the learned model has something to beat, and both numbers
go on the slide. That is the argument, not caution: it proves the ML earns its place.

### Why SAR

Floods happen during the monsoon; during the monsoon India is under complete cloud.
Optical sees nothing. SAR penetrates cloud. It is the only imagery that works on the day
it matters.

### Tiled Otsu, and the failure it hides

Over a whole district the flooded fraction is usually small, so the global histogram is
unimodal and Otsu returns nonsense. We tile, score each tile for genuine bimodality, and
take the median threshold of the best tiles.

**This has a nasty failure mode that we hit during development.** A tile of *pure land*
is often bimodal for a different reason — dark bare fields against radar-bright built-up
area, which double-bounces off walls. Otsu happily splits that, the tile scores as
bimodal, and the threshold lands around −10 dB, classifying every dry field as flood.
Unchecked, the system reported **more** inundation at low water than at high.

The fix is a physical ceiling: open water at C-band VV sits below roughly −15 dB, so any
threshold above −13 dB is rejected as a land/land split. `test_hazard.py` has the
regression test, and `test_integration.py` asserts extent is monotonic in river stage.

For uncalibrated input (raw GRD digital numbers, which run *positive*), absolute
thresholds are meaningless, so the ceiling is derived from the scene's own histogram
instead — and the result is labelled as uncalibrated all the way to the API.

### Extent → depth, via HAND

Satellites give extent. Evacuation needs depth: 0.1 m is a puddle, 0.4 m is a vehicle
trap, 0.7 m kills pedestrians.

> Water finds its level. At the **edge** of a flood the depth is zero — that is what makes
> it the edge. So the HAND value at the edge is the water-surface height for that patch
> of floodplain, and any pixel inside is submerged by the difference between that surface
> and its own HAND.

`depth = water_level_HAND − HAND(pixel)`

The water level is taken **locally**, not once per flood body, because a 40 km river reach
is not level end to end — one level per component would over-estimate depth upstream and
under-estimate it downstream.

**Why HAND rather than 2-D hydrodynamics:** C-FLOOD covers three basins after years of
work because full hydrodynamics is expensive to set up per basin. HAND is coarser but
generalises to any district with a DEM. That is the right trade when the decision you are
supporting is "which road, which shelter" — not millimetre depth.

### The flat-floodplain problem

The Brahmaputra floodplain is almost perfectly flat. After depression filling, huge
regions share one elevation and D8 has no downslope neighbour to choose, so flow routing
produces parallel garbage and HAND becomes meaningless.

`resolve_flats()` imposes a tiny gradient running away from higher ground and toward the
flat's outlets, built from two distance transforms. This step is not optional here.

The channel-initiation threshold is expressed as **contributing area in km², not cell
count** — 1000 cells means 0.9 km² at 30 m and 8.1 km² at 90 m, so a cell-count threshold
silently changes meaning with resolution and drags every depth in the system with it.

---

## Block 3 — Impact

Three spatial joins. No ML, and we do not pretend otherwise — bolting a model onto a
problem a lookup table solves is something a panel with a real ML person will punish.

**Controlling depth is the 90th percentile along an edge, not the mean.** A road is only
as passable as its worst point; averaging a 300 m causeway with one flooded dip produces a
comfortable number and a drowned driver. A percentile rather than the maximum so one
misclassified pixel cannot close a highway.

**Caution is a speed penalty, not a cut.** Movement through shallow water is slower, not
impossible.

**Evacuation demand is not population.** Handing the optimiser every person in the
district asks it to evacuate 1.7 million people into 50,000 shelter places and buries the
real answer. Flooded population pixels are attributed to their nearest settlement — a
Voronoi partition of the floodplain — and that becomes the demand. This was a genuine
modelling error caught during development.

---

## Block 4 — Decision ★

### Order of operations

```
1. assess shelters      a flooded shelter must be removed BEFORE travel times are
                        computed, or the optimiser spends its whole run routing people
                        to a swimming pool
2. annotate roads       depth → passability → per-edge travel cost
3. prune                delete impassable edges
4. bind                 attach settlements and shelters to the network
5. travel-time matrix   ONE Dijkstra PER SHELTER on the reversed graph
6. assign               min-cost flow under capacity constraints
7. route                shortest paths + the flood-blind counterfactual
8. prioritise           whoever the plan cannot serve becomes the boat list
```

Getting 1 before 5 wrong produces a plan that looks complete and is dangerous.

### Travel times

One Dijkstra per **shelter**, on the reversed graph — not one per settlement. There are
far fewer shelters than settlements, and reversing turns "distance from all settlements
to this shelter" into a single-source problem.

Unreachable pairs are simply **absent** from the matrix. That absence is the signal the
priority ranking consumes: a settlement with no entries at all is cut off.

### The optimisation

**Read `decision/assignment.py` before presenting it.**

Nearest-shelter is not a decision, it is a lookup, and it breaks the moment two villages
share a nearest shelter. The correct question is: *what allocation minimises total travel
time subject to no shelter exceeding capacity?* That is a transportation problem, solved
exactly by minimum-cost flow.

```
settlement_i  demand = −people_at_risk_i
shelter_j     demand = 0
SINK          demand = +total
UNSHELTERED   demand = 0

settlement_i → shelter_j     cap = demand_i,   cost = travel_seconds
shelter_j    → SINK          cap = capacity_j, cost = 0
settlement_i → UNSHELTERED   cap = demand_i,   cost = PENALTY
UNSHELTERED  → SINK          cap = ∞,          cost = 0
```

Flow conservation forces everyone out of their settlement; the only exits are through a
capacity-bounded shelter or through `UNSHELTERED`. **The capacity constraint is
structural, not a post-hoc check** — the solver cannot overfill a shelter.

`UNSHELTERED` exists because without it the problem is *infeasible* whenever population
exceeds capacity, and in Assam it routinely does. A system that refuses to answer in
exactly the situation it was built for is useless. The penalty is set far above any
achievable travel cost, making the objective lexicographic: maximise people sheltered
first, minimise travel time second. "Infeasible" becomes a number — and that number is the
ask a district officer takes to the state.

### The counterfactual

Routes are computed **twice**: once on the pruned graph, once on the unpruned one costed
by length — which is what a flood-blind navigation app would return. Where they differ, we
can name the specific road we steered away from.

That is what generates *"take the road via Jania — not the main road, it is under water"*
from data rather than from a slide.

---

## Block 5 — Delivery

Three audiences, three genuinely different artefacts:

| Audience | Artefact | Design constraint |
|---|---|---|
| District officer / NDRF | Dashboard | Dark UI (a control room at night), attributes clickable |
| Citizen | Action card | One screen, one destination, one route, one explicit negative instruction |
| Machines | CAP 1.2 XML | Element order is an `xsd:sequence`; polygons are `lat,lon` |

**Depth is sent as a PNG overlay, roads as GeoJSON.** Vectorising a 2-megapixel depth
raster yields tens of thousands of ragged polygons and renders slowly; a georeferenced PNG
is a few hundred KB and preserves the gradient. Roads stay vector because their
*attributes* are the point — clicking a red segment must show "0.42 m, impassable".

**Action cards use pre-translated templates, not an LLM at request time.** An emergency
message must be deterministic; there must be no network dependency at the moment of use;
and translations can only be reviewed in advance if they exist in advance.

---

## Block 6 — Demo harness

`LIVE` (today's feeds) · `REPLAY` (a historical event, stepped) · `SYNTHETIC` (procedural,
zero downloads).

Static layers are shared **by reference** across replay frames, so a 48-hour replay costs
48 hazard rasters, not 48 copies of the DEM.

The synthetic district is not a toy — it reproduces the things that matter: a braided
meandering channel, natural levees (ground beside a channel is *higher* than the floodplain
behind it), only four bridges, char island villages with at most two road links, and
radar-bright built-up areas that SAR under-detects. Without char villages the district
never fragments, and the boat dispatch list — the whole point — never populates.

---

## Honesty as an architectural property

Not a disclaimer section. Enforced in code:

- `Provenance` records ride on the `Scenario` object; the dashboard banner is **generated
  from them**, so it cannot show a clean screen over synthetic data.
- `CAPConfig` **refuses** `status="Actual"` without an explicit environment override.
- Unobserved area is tracked as a mask, and flooded fraction is computed over the
  **observed** area, never the whole grid.
- `compare_to_baseline()` requires a ≥2-point IoU gain before the U-Net is allowed to ship.
- Estimated shelter capacity is a typed field (`capacity_source`), surfaced in every API
  response and rendered differently in the UI.

The intent is that the path of least resistance produces an honest interface, and a
dishonest one takes extra work.
