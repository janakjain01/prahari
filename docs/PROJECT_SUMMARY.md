# PRAHARI — Project Summary

**Smart India Hackathon 2026 · Disaster Management · Barpeta district, Assam**

A complete summary of what this project is, what it does, everything that was built,
and what has actually been verified to work.

---

## 1. The goal

> **India can already predict floods. It still cannot tell a family which road is safe.**

Every layer of flood *forecasting* already exists in India, and most of it is free.
What does not exist anywhere is the **decision layer** — the step that turns a forecast
into an instruction a specific family in a specific village can act on.

| System | What it does | Where it stops |
|---|---|---|
| **SACHET** (NDMA + C-DOT) | Alert delivery — cell broadcast, CAP geofencing, local languages | It is a *pipe*. It says a flood is coming, not what to do |
| **CWC** | River level and inflow forecasts | Gauged stations and towns — not every village |
| **C-FLOOD** (C-DAC + NRSC + CWC) | Real 2-D hydrodynamic flood maps | Only 3 basins: Mahanadi, Godavari, Tapi. **Brahmaputra and Kosi are not covered** |
| **Google Flood Hub** | Best-in-class AI riverine forecasting | Riverine only. No routing, no shelter data, no last-mile action |

PRAHARI's goal is deliberately narrow and defensible: **consume everyone else's forecasts,
and compute the one thing none of them produce — which road this village should take, to a
shelter that still has room.**

We are not competing with ISRO on hydrology. That positioning is the project's strongest
defensive asset in front of a panel.

---

## 2. What it actually does

Given a flood state (from satellite or forecast), PRAHARI produces six outputs:

| # | Output | Detail |
|---|---|---|
| 1 | **Which roads are cut** | Per-segment water depth and a passability class, from `clear` through `life_threatening` |
| 2 | **Where everyone goes** | Capacity-constrained shelter assignment solved as a minimum-cost flow |
| 3 | **How they get there** | Safe route on a road graph with flooded edges deleted |
| 4 | **Who cannot get out** | Ranked **boat dispatch list** — settlements with no road route at all |
| 5 | **What to tell them** | One-screen citizen action card in Assamese / Bengali / Hindi / English |
| 6 | **How it reaches them** | CAP 1.2 XML — the same schema SACHET already runs, which drives cell broadcast to feature phones |

**The line that carries the pitch:**

> Google Maps will route you straight into the water. We route you around it —
> and we make sure the shelter we send you to still has room.

---

## 3. Everything that was implemented

### 3.1 Spatial core

| Module | What it does |
|---|---|
| `grid.py` | Metric raster grid. **Every layer in the system lives on one grid in one projected CRS with metre units.** Removes resampling mismatches and degrees-vs-metres hydrology bugs by construction. Depends on numpy alone |
| `scenario.py` | The object that flows through the whole pipeline. Separates **static** layers (terrain, roads, people — expensive, never change) from **dynamic** layers (flood, depth — cheap, recomputed each cycle). Carries `Provenance` records on every layer |
| `config.py` | Typed, validated district configuration. **No magic numbers anywhere else in the codebase** — every threshold a judge might question lives in one YAML file with a citation |

### 3.2 Hazard layer — where is the water, how deep

| Module | What it does |
|---|---|
| `hazard/hand.py` | Full HAND hydrology: **Priority-Flood + ε** conditioning (fills depressions *and* resolves flats with a guarantee of zero sinks — essential, because the Brahmaputra floodplain is nearly flat and destroys naive D8), D8 flow routing, flow accumulation, drainage network, Height Above Nearest Drainage |
| `hazard/otsu.py` | Classical SAR water detection — the **baseline the U-Net must beat**. Lee speckle filter, tiled Otsu with bimodality scoring, physical open-water ceiling, terrain veto, blob cleanup, per-pixel confidence |
| `hazard/depth.py` | Flood **extent → depth** via HAND. Spatially-varying water surface interpolated from flood edges, because a 40 km river reach is not level end to end |
| `hazard/unet.py` | U-Net with pretrained ResNet-34 encoder, adapted to 2-channel SAR input. Dice + BCE loss with ignore-mask support |
| `hazard/dataset.py` | Sen1Floods11 loader with correct `-1 = unlabelled` handling, plus a synthetic stand-in so training code stays runnable offline |
| `hazard/train.py` | Training loop that **evaluates the Otsu baseline on the same test split in the same run**, and refuses to mark a checkpoint shippable if it trained on synthetic data |
| `hazard/infer.py` | Single entry point. Picks U-Net if a checkpoint exists and beat the baseline, otherwise the baseline. Tiled inference with cosine blending (naive tiling produces seams that look like embankments) |
| `hazard/metrics.py` | IoU / F1 / precision / recall, and a **ship / no-ship verdict** requiring ≥0.02 IoU gain |

### 3.3 Impact layer — flood × roads × people × shelters

| Module | What it does |
|---|---|
| `impact/roads.py` | Depth sampled along every road edge; **controlling depth is the 90th percentile, not the mean** — a road is only as passable as its worst point. Separate vehicle and pedestrian costs |
| `impact/population.py` | Exposure aggregated to 1 km reporting cells; settlement clustering from the population raster; **`compute_evacuation_demand()`** — Voronoi attribution of flooded population to settlements |
| `impact/shelters.py` | Shelter viability (a shelter in floodwater is not a shelter), capacity estimation from building footprint with an explicit plinth-height allowance |

### 3.4 Decision layer ★ — the differentiator

| Module | What it does |
|---|---|
| `decision/graph.py` | Prunes impassable edges. **Keeps isolated nodes deliberately** — a stranded node is a cut-off settlement, not litter. KD-tree feature binding that leaves far features *unbound* rather than snapping them across country |
| `decision/assignment.py` | **Minimum-cost flow shelter assignment.** Capacity is a structural constraint, not a post-hoc check. An `UNSHELTERED` overflow node makes the problem always feasible and turns "infeasible" into a number |
| `decision/routing.py` | Dijkstra on the pruned graph, **plus the counterfactual** — the route a flood-blind navigation app would give — so we can name the specific road we steered around |
| `decision/priority.py` | Ranked dispatch list with five reason codes. Separates `CUT_OFF` (boat tasking tonight) from `UNSHELTERED` (capacity ask to the state) |
| `decision/plan.py` | Orchestrator. Order of operations matters: shelters are assessed **before** travel times, or the optimiser spends its run routing people to a swimming pool |

### 3.5 Delivery layer

| Module | What it does |
|---|---|
| `delivery/cap.py` | **CAP 1.2 XML** with correct `xsd:sequence` element order and `lat,lon` polygons (the opposite of GeoJSON). Includes a self-validator |
| `delivery/action_card.py` | Citizen cards: one screen, one destination, one route, one explicit negative instruction. **Stranded settlements get a different card**, not a softened one |
| `delivery/i18n.py` | Pre-translated templates in Assamese, Bengali, Hindi, English. **Deliberately not LLM-generated at request time** — a safety message must be deterministic and must work offline |
| `delivery/dashboard.py` | GeoJSON layers, depth PNG overlay with a colour ramp keyed to **decision thresholds** rather than an aesthetic gradient, and a disclosure block generated from provenance |

### 3.6 Ingestion — real data, all optional

| Module | Source | Key needed? |
|---|---|---|
| `ingest/dem.py` | Copernicus DEM GLO-30 + OSM embankment burning | no |
| `ingest/sentinel1.py` | Sentinel-1 RTC/GRD via Planetary Computer STAC | **no** |
| `ingest/osm.py` | Roads, shelter candidates, embankments | no |
| `ingest/population.py` | WorldPop 2020, with download-to-cache fallback | no |
| `ingest/weather.py` | Open-Meteo; Google Flood Hub and IMD *optional* | mixed |
| `ingest/build.py` | Assembles a full real scenario with a per-layer report | — |

**Every adapter returns `IngestResult(ok=False, reason=…)` rather than raising.** No feed
is load-bearing — the two we'd most like (IMD, Google Flood Hub) are *approval queues*,
not tasks, so nothing depends on them.

### 3.7 Validation

| Module | What it does |
|---|---|
| `validate/permanent_water.py` | **Separates flood from river** using JRC Global Surface Water. A correctness fix first — without it, we'd report the Brahmaputra as flooding |
| `validate/retrospective.py` | Runs the pipeline across a real event's full Sentinel-1 series and checks the flood **rises, peaks and recedes**. Optional IoU scoring against a supplied reference mask |

### 3.8 Demo harness

| Module | What it does |
|---|---|
| `demo/synth.py` | Procedural Brahmaputra floodplain: braided meandering channel, **natural levees**, backswamps, only 4 bridges, **char island villages** with ≤2 road links, radar-bright built-up areas. Renders synthetic SAR with speckle |
| `demo/replay.py` | Steps a historical event hour by hour. Static layers shared by reference, so 48 frames cost 48 hazard rasters, not 48 DEMs |

### 3.9 Interface

| Component | What it does |
|---|---|
| `api/main.py` | FastAPI: `/api/state`, `/api/overlay.png`, `/api/cap.xml`, `/api/cap/validate`, `/api/cards`, `/api/priorities.csv`, `/api/replay` |
| `api/state.py` | Engine with disk-cached static layers and a bounded plan cache |
| `web/` | **Zero-build MapLibre dashboard.** No npm, no framework. MapLibre vendored locally — works with no internet at all |
| `cli.py` | `serve`, `demo`, `real`, `ingest`, `validate`, `cap`, `status`, `fetch-population` |

---

## 4. Workability — what has actually been verified

This section is the honest answer to "does it work?"

### 4.0 The hazard model — trained, and it earned its place

Trained on the real Sen1Floods11 hand-labelled split (252 train / 89 val / 90 test),
downloaded from its public bucket with no account or API key. Evaluated on the
**official held-out test split**, against the Otsu baseline **in the same run**:

| | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| **U-Net (ResNet-34)** | **0.6733** | 0.8048 | 0.8260 | 0.7846 |
| Otsu baseline | 0.4988 | 0.6656 | 0.7217 | 0.6175 |
| **Gain** | **+0.1745** | +0.139 | +0.104 | +0.167 |

The ship/no-ship gate requires ≥0.02 IoU. **Verdict: the U-Net ships.**

A first run stopped early at epoch 10 with its best at epoch 2 — the 89-tile validation
split makes val IoU noisy, and a patience of 8 misread that oscillation as a plateau,
completing only a quarter of the LR schedule. Retrained with a longer patience: best
epoch 22, test IoU 0.6543 → **0.6733**. Both runs are kept in `models/`.

**But see §5, Bug 5** — the model is *not* the default on real Indian scenes, and the
reason matters.

### 4.1 Test suite

```
85 tests · fully offline · ~17 seconds · all passing
```

Covering the decision core (capacity never exceeded, nobody vanishes, min-cost flow beats
greedy), hydrology (flow conservation, HAND monotonicity), detection, CAP schema
compliance, translations, and full API integration.

Includes **regression tests for the two worst bugs found during development** — both
invisible to unit tests and obvious the moment the whole chain ran.

### 4.2 Verified on real data — every layer

The complete pipeline runs against the **real July 2024 Barpeta flood**, with
**no synthetic inputs at all**:

| Layer | Result |
|---|---|
| Copernicus DEM GLO-30 | ✅ 2 tiles, 100% coverage, 28.4–182.6 m |
| HAND hydrology | ✅ 0.38% of cells are channel; max flow accumulation 1.26 M cells |
| OSM road network | ✅ 3,696 edges |
| OSM shelters | ✅ 14 candidates *(sparse — a real OSM gap)* |
| OSM embankments | ✅ 1 centreline *(also a real gap)* |
| **WorldPop 2020** | ✅ **488 MB downloaded and cached — real population** |
| **Settlements** | ✅ **116 clustered from real population** |
| Sentinel-1 RTC, 16 July 2024 | ✅ calibrated, 69% district coverage |
| Flood detection | ✅ −13.90 dB, 25.6% of observed area flooded, max depth 7.92 m |
| Full decision plan | ✅ built in ~1 s (39 s end to end including all fetches) |
| CAP 1.2 output | ✅ structurally valid |

**Real output for 16 July 2024:**

| Metric | Value |
|---|---|
| District population (WorldPop, within the AOI) | 1,400,221 |
| People inside the flood footprint | 223,091 |
| **People needing evacuation** | **223,091** *(reconciles exactly — nobody is dropped)* |
| Sheltered by the plan | 37,404 |
| **Capacity shortfall** | **185,685** |
| **Settlements cut off (boat dispatch)** | **4 — 10,726 people** |
| Roads cut | 12.1 km |
| Flooded area | 320.5 km² (25.6% of the observed swath) |

The capacity shortfall is enormous because **OSM has only 14 shelters mapped for a
district of 1.4 million people**. That is a real data gap, reported honestly rather than
papered over — and it is precisely the number that would go to the state asking for the
ASDMA relief-camp register.

Sample boat dispatch output, with navigable names derived from OSM places:

```
#9   8 km W of Mandia         6,874 at risk   26.2559, 90.8872
#23  4 km SE of Kalgachia     2,144 at risk   26.3304, 90.8938
#29  7 km W of Mandia         1,672 at risk   26.2550, 90.8980
#62  6 km N of Sarupeta          36 at risk   26.5489, 91.0753
```

**No API key. No waitlist. Free data, today.**

### 4.3 Retrospective validation — run, not just built

`prahari validate` was executed against four real Sentinel-1 acquisitions around the
July 2024 event:

| Date | Coverage | Flood km² | River km² | Exposed | Cut off |
|---|---|---|---|---|---|
| 2024-06-22 | 69% | 338.3 | 12.0 | 251,534 | 5 (9,061) |
| **2024-06-24** | **100%** | **498.4** ← peak | 14.2 | 268,245 | 8 (16,830) |
| 2024-06-29 | 100% | 415.4 | 13.6 | 234,920 | **9 (15,628)** |
| 2024-07-04 | 69% | 376.8 | 15.8 | 270,020 | 5 (12,824) |

**The flood extent rises to a peak and recedes** — the series behaves like a real
hydrograph, which is exactly what a working detector should produce over a real event and
what a broken one cannot fake. Permanent water (JRC GSW) averaged 14 km² and was
subtracted, so these are flood figures and not the Brahmaputra.

At peak, **9 settlements holding 15,628 people had no road route to any viable shelter.**

Full report: `data/out/validation_barpeta_2024-07-16.{json,md}`

### 4.4 Generalisation — proven, not asserted

The same pipeline was run on **Darbhanga, Bihar** (Kosi-Bagmati basin) by swapping one
YAML file. Nothing in the codebase is specific to Assam.

| Layer | Barpeta, Assam | Darbhanga, Bihar |
|---|---|---|
| Road edges | 3,696 | **73,507** |
| Shelters (OSM) | 14 | **92** |
| Named places (OSM) | 12 | **86** |
| Embankments mapped | 1 | **45** |
| Population (WorldPop) | 1,400,221 | 3,606,760 |
| Settlements derived | 116 | 136 (49 with exact OSM names) |
| Sentinel-1 swath coverage | 69% | **100%** |

Darbhanga is comfortably the better *demo*, because OSM coverage in Bihar is far richer.
Barpeta is the better *argument*, because it is in the basin C-FLOOD does not cover.

On 17 July 2024 Darbhanga showed only 0.7% inundation — which is correct rather than a
failure: the July 2024 flood was in Assam, not Bihar. A system that reported a
catastrophe there would be broken.

### 4.5 Performance

| Operation | Time |
|---|---|
| Static layers (cached) | ~10 s |
| Static layers (cold, real data) | ~26 s |
| Full decision recompute per flood state | **~1 second** |
| Full test suite | 17 s |

The ~1 s recompute is why the replay scrubber is instant on stage.

### 4.6 Behaviour verified across water levels

| River stage | Flooded | Needing evacuation | Sheltered | Shortfall | Cut off |
|---|---|---|---|---|---|
| 1.0 m | 10% | 50,006 | 49,833 | 174 | 2 |
| 2.0 m | 18% | 101,851 | 59,446 | 42,406 | 4 |
| 3.0 m | 27% | 162,715 | 56,411 | 106,301 | 4 |
| 4.0 m | 39% | 321,304 | 53,138 | 268,164 | 5 |
| 5.0 m | 48% | 423,907 | 50,973 | 372,938 | 6 |

Monotonic and physically coherent throughout — which is exactly the property that failed
before the tiled-Otsu bug was found and fixed.

---

## 5. Five significant bugs found and fixed

Worth knowing, because they demonstrate the system was genuinely exercised on real data
rather than merely written. All three were **silent** — nothing crashed.

### Bug 0 — the hydrology collapsed on the real DEM *(the worst one)*

Depression filling left **43.8%** of the real Barpeta DEM as perfectly flat cells. The
flat-resolution step was supposed to give those a drainage gradient, and it did not work:
it measured distance to the nearest draining cell **geometrically, ignoring elevation**, so
on a floodplain the gradient frequently pointed uphill across a ridge.

Consequences, none of which raised an error:

- 15.7% of the district stayed as sinks;
- the flow network shattered into ~320,000 one-cell basins;
- maximum flow accumulation was **595 cells out of 2,000,000**;
- no cell ever reached the channel threshold, so HAND silently fell back to
  "the lowest 1% of terrain";
- **every depth in the system was wrong**, and therefore every road-cut decision.

The symptom that exposed it: HAND was **identical to two decimal places across an 80× range
of drainage thresholds**. A parameter that does nothing is a parameter that isn't wired up.

**Fix:** replaced fill + separate flat resolution with **Priority-Flood + ε**
(Barnes, Lehman & Mulla 2014), which fills and resolves flats in one pass with a
guarantee — every cell is assigned a height strictly above the cell it was reached from,
so a strictly-descending path to the boundary exists by construction.

**Result:** interior sinks 15.7% → **0**. Max accumulation 595 → **1,264,212 cells** (62% of
the district draining through one outlet, i.e. an actual river). 11.8 s, one-time, cached.
Now covered by four regression tests, including one asserting HAND *must* respond to the
drainage threshold.

### Bug 1 — the detector reported *more* flooding at low water than at high

Tiled Otsu selects tiles whose histogram is bimodal. But a tile of **pure dry land** is
often bimodal for a completely different reason: dark bare fields against radar-bright
**built-up** area, which double-bounces radar off walls. Otsu split *those*, the tile
scored as bimodal, and the derived threshold landed around −10 dB — classifying every dry
field in the district as flood.

Frame 0 of the replay (stage 0.4 m) reported **294,928** people needing evacuation; stage
3.0 m reported 162,715. Backwards.

**Fix:** a physical ceiling. Open water at C-band VV sits below ~−15 dB, so any threshold
above −13 dB is rejected as a land/land split. For uncalibrated input the ceiling is
derived from the scene's own histogram instead. Now covered by both a unit regression test
and an integration test asserting monotonicity in river stage.

### Bug 2 — the dashboard blanked entirely when WebGL failed

Every render was gated inside `map.on('load')`. Any WebGL failure — blocklisted GPU,
locked-down projector laptop, remote desktop, a backgrounded tab — left the **entire
screen blank**, including the dispatch list and every number, none of which need a map.

**Fix:** data rendering fully decoupled from the map. Panels render immediately; map layers
are pushed separately when (or if) the map becomes ready, with a 12-second timeout and an
explicit fallback notice. On demo day this converts a total loss into a degraded one.

Also fixed along the way: an evacuation-demand modelling error (the optimiser was
evacuating the entire 1.7 M district rather than the people actually at risk), a
resolution-dependent drainage threshold, a DEM embankment-burn that stacked height on
repeated cells, a `classList.add('')` crash, NaN handling in OSM tag parsing, and a
`CPL_VSIL_CURL_USE_HEAD=NO` setting that stopped GDAL discovering HTTP range support and
so blocked the WorldPop fetch entirely.

### Bug 4 — most flooded people were attributed to nobody

Settlements are clustered from the population raster, which picks out dense cores. On the
real WorldPop raster those cores held only **377k of the district's 1.4 M people**; the
rest live dispersed between them. Evacuation demand was then capped at each cluster's core
population, so people living between clusters were counted as *exposed* but never as
*needing evacuation*.

The tell: `exposed_population = 223,091` but `needing_evacuation = 49,124`. **174,000
people were silently dropped from the plan.**

**Fix:** a settlement point now represents its **catchment**, not its core. Every populated
pixel is attributed to its nearest settlement by Voronoi partition, so settlement totals
reconcile with the district population. The two figures now match exactly, and the
cut-off count rose from 2,062 to 10,726 once catchments were counted properly.

### Bug 5 — the trained model disagrees with the baseline on real Indian scenes

On the real Barpeta scene of 16 July 2024, the two detectors differ by **2.5×**:
Otsu 320.5 km², U-Net 130.7 km². The U-Net wins on Sen1Floods11 by a clear margin, so
this is not simply "the model is worse".

Measuring the two distributions found the likely cause — **domain shift**:

| | VV mean | VV std | VV p99 |
|---|---|---|---|
| Sen1Floods11 (measured) | −9.69 dB | **3.43** | −3.8 dB |
| Barpeta RTC scene | −10.95 dB | **6.21** | **+1.6 dB** |

Nearly double the spread and far brighter extremes. Sen1Floods11 is 11 global events as
GRD; we infer on RTC over a braided Himalayan floodplain.

Two things were done, and neither is a claim that the problem is solved:

1. **Real scenes default to the Otsu baseline.** It is physically grounded,
   terrain-vetoed, explainable on a whiteboard, and errs toward *more* water — the right
   direction to be wrong in during an evacuation. The U-Net stays available via
   `?detector=unet`.
2. **Normalisation statistics now travel inside the checkpoint**, so a train/inference
   mismatch cannot silently degrade predictions. While investigating, the hardcoded
   constants turned out never to have matched the real training data (−12.0/5.0 assumed
   vs −9.69/3.43 measured).

**Neither flood-extent figure should be quoted as fact** until both are scored against an
independent reference mask, which `prahari validate --reference` already supports.

**The pattern worth noting:** every one of these was invisible to unit tests and only
appeared when the whole chain ran against real observations. That is the argument for
running on real data early rather than polishing a synthetic demo.

---

## 6. The honest split — ML vs engineering

| Component | Type | Why |
|---|---|---|
| SAR flood segmentation | **Real ML** — U-Net on Sen1Floods11 | Genuine CV problem, public labelled data |
| HAND depth estimation | Deterministic hydrology | Established method (Nobre et al. 2011). Not ML, and we say so |
| Road passability | Rules + spatial join | A lookup table is *better* here than a model |
| **Shelter assignment** | **Optimisation — min-cost flow** | The smartest part. Operations research, not ML |
| Routing | Dijkstra on a pruned graph | The correct tool |
| Multilingual cards | Pre-translated templates | A safety message must not be paraphrased |

**One serious model, one exact optimiser, and a lot of careful engineering.** We did not
add models to look impressive. A panel with a real ML person will respect *"we used a rule
here because a model would be worse"* far more than a network predicting something a
lookup handles.

---

## 7. Honesty built into the product

This is an architectural property, enforced in code — not a disclaimer section:

- **The dashboard cannot show a clean screen over synthetic data.** The banner is generated
  from the scenario's own `Provenance` records, not from a flag someone might forget.
- **Estimated shelter capacity renders differently from surveyed capacity**, everywhere.
- **Unobserved area is never reported as dry.** A Sentinel-1 pass covers 69% of Barpeta;
  the rest is *unknown*, and extent is explicitly a lower bound.
- **Permanent water is subtracted**, so we report flooding, not the Brahmaputra.
- **CAP `status` cannot be set to `Actual`** without an explicit environment override.
- **The U-Net only ships if it beats the baseline** by ≥0.02 IoU, with both numbers shown.
- **Training refuses to mark a checkpoint shippable** if it trained on synthetic data.

---

## 8. Known limitations

The sharpest ones, stated before a judge finds them (full list in `docs/LIMITATIONS.md`):

1. **A 30 m DEM cannot resolve embankments** — sub-pixel in both dimensions, and breached
   embankments are a primary flood mechanism in Barpeta. Partially mitigated by burning OSM
   dyke lines; OSM found only **one** for the whole district.
2. **Shelter capacity is always estimated**, never surveyed. Needs the ASDMA relief-camp register.
3. **OSM coverage is uneven** — a settlement can look cut off because its road was never
   mapped. Handled with a separate `UNBOUND` reason code rather than being hidden.
4. **Depth is modelled and unvalidated** — no public per-pixel depth truth exists.
5. **SAR under-detects urban flooding** — buildings double-bounce radar back to the sensor.
6. **The evacuation plan itself is unvalidated** — nobody recorded which road each family took.
7. **Translations are not native-speaker validated.** This is a hard prerequisite for deployment.

Most remaining work is **institutional access, not engineering**.

---

## 9. How to run it

```bash
pip install -r requirements.txt

python -m prahari.cli serve --downsample 2      # dashboard at http://127.0.0.1:8000
python -m prahari.cli demo --stage 3.0          # synthetic, no network needed
python -m prahari.cli real --date 2024-07-16    # REAL Sentinel-1 + DEM + OSM
python -m prahari.cli validate --date 2024-07-16
python -m prahari.cli status                    # what is configured, what is not
python -m pytest tests/ -q                      # 85 tests, ~17 s
```

**Scaling to a second district is one YAML file** — `config/darbhanga.yaml` runs the same
pipeline on the Kosi-Bagmati in Bihar. Nothing is hardcoded to Assam.

---

## 10. Documentation map

| Document | Purpose |
|---|---|
| `README.md` | Entry point and quick start |
| `docs/ARCHITECTURE.md` | How it works and why each decision was made |
| `docs/REFERENCES.md` | **Where every threshold comes from.** Read before presenting |
| `docs/JUDGE_QA.md` | The 15 questions you will be asked, with answers |
| `docs/VALIDATION.md` | What is validated, what is not, how to check |
| `docs/LIMITATIONS.md` | What this cannot do |
| `docs/DEMO_RUNBOOK.md` | Demo day: what to run, in what order, what to say |

---

## 11. Status

**Working prototype.** Runs end to end on real observations for Barpeta, Assam, with no API
keys and no waitlisted feeds. 85 tests passing offline.

**Not an operational system.** Nothing here is certified or approved by NDMA, ASDMA or IMD,
and no part has been tested with people who would rely on it.

---

### The one-liner

> India can already predict floods. It still can't tell a family which road is safe.
> That's the gap. That's all we built.
