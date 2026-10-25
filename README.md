# PRAHARI — Flood Evacuation Decision Layer

**India can already predict floods. It still cannot tell a family which road is safe.
PRAHARI is the decision layer that turns a flood warning into a route.**

SIH 2026 · Disaster Management · Barpeta district, Assam (Brahmaputra basin)

---

## The gap, in one table

| System | What it does | Where it stops |
|---|---|---|
| **SACHET** (NDMA + C-DOT) | Alert delivery — cell broadcast, CAP geofencing, local languages | It is a *pipe*. It tells you a flood is coming, not what to do |
| **CWC** flood forecasting | River level and inflow forecasts | Gauged stations and towns — not every village |
| **C-FLOOD** (C-DAC + NRSC + CWC) | Real 2-D hydrodynamic flood maps | Three basins: Mahanadi, Godavari, Tapi. **Brahmaputra and Kosi are not covered** |
| **Google Flood Hub** | Best-in-class AI riverine forecasting, free | Riverine only. No evacuation routing, no shelter data, no last-mile action |

> Every layer of forecasting exists and most of it is free.
> **The decision layer does not exist at all.**

We are not competing with ISRO on hydrology. We consume everyone's forecasts and do
the one thing none of them do: **compute which road a specific village should take,
to a shelter that still has room.**

---

## What it actually produces

```
1. WHICH ROADS ARE CUT      per-segment water depth and a passability class
2. WHERE EVERYONE GOES      capacity-constrained shelter assignment (min-cost flow)
3. HOW THEY GET THERE       safe route on a graph with the flooded edges deleted
4. WHO CANNOT GET OUT       ranked boat dispatch list — the output nobody else produces
5. WHAT TO TELL THEM        one-screen action card in Assamese / Bengali / Hindi / English
6. HOW IT REACHES THEM      CAP 1.2 XML — the same schema SACHET already runs on
```

**The line that wins the room:**
> Google Maps will route you straight into the water. We route you around it —
> and we make sure the shelter we send you to still has room.

---

## Quick start

```bash
pip install -r requirements.txt
```

Run the dashboard (no network, no API keys, no downloads):

```bash
python -m prahari.cli serve --downsample 2
```

Then open <http://127.0.0.1:8000>.

Run the whole pipeline in the terminal instead:

```bash
python -m prahari.cli demo --district barpeta --stage 3.0
```

Run it on **real observations** — real DEM, real Sentinel-1, real OSM:

```bash
python -m prahari.cli real --district barpeta --date 2024-07-16
```

Validate against a real historical flood:

```bash
python -m prahari.cli validate --district barpeta --date 2024-07-16
```

Train the hazard model (Sen1Floods11 is a public bucket — no account, no key):

```bash
python -m prahari.cli fetch-training-data
```

```bash
python -m prahari.cli train --epochs 30
```

Check what is configured and what is missing:

```bash
python -m prahari.cli status
```

---

## This runs on real data, today

Verified working against the **July 2024 Barpeta flood**, with no API key and no waitlist.
Retrospective validation over four real Sentinel-1 acquisitions shows flood extent rising
to a **498 km² peak on 24 June and receding** — the series behaves like a real hydrograph.
At peak, **9 settlements holding 15,628 people had no road route to any viable shelter.**
See [docs/VALIDATION.md](docs/VALIDATION.md).

| Layer | Source | Key needed? |
|---|---|---|
| Elevation | Copernicus DEM GLO-30 (Planetary Computer) | no |
| SAR imagery | Sentinel-1 RTC (Planetary Computer) | no |
| Roads, shelters, embankments | OpenStreetMap | no |
| Permanent water | JRC Global Surface Water | no |
| Rainfall | Open-Meteo | no |
| Population | WorldPop 2020 | no |
| *Riverine forecast* | *Google Flood Hub* | **waitlisted — optional by design** |
| *District warnings* | *IMD API* | **registration + IP whitelist — optional by design** |

The two gated feeds are **approval queues, not tasks**. No amount of effort makes them
arrive faster, so nothing is built to depend on them. Apply on day one; assume neither lands.

---

## Architecture

```
BLOCK 1 · DATA SPINE      DEM · OSM roads · population · shelters · SAR · rainfall
        ↓
BLOCK 2 · HAZARD          Sentinel-1 → water mask (Otsu baseline / U-Net)
                          → depth via HAND                              ← ML lives here
        ↓
BLOCK 3 · IMPACT          flood ∩ roads · flood ∩ people · flood ∩ shelters
        ↓
BLOCK 4 · DECISION        prune flooded edges → min-cost flow assignment
                          → routing → cut-off ranking                   ★ THE DIFFERENTIATOR
        ↓
BLOCK 5 · DELIVERY        responder dashboard · citizen card · CAP 1.2 export
        ↓
BLOCK 6 · DEMO HARNESS    live mode · replay mode · synthetic mode
```

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### The hazard model

Trained on the real Sen1Floods11 hand-labelled split, scored on the official held-out
test set against the classical baseline **in the same run**:

| | IoU | F1 | Recall |
|---|---|---|---|
| **U-Net (ResNet-34)** | **0.6733** | 0.8048 | 0.7846 |
| Otsu baseline | 0.4988 | 0.6656 | 0.6175 |

The ship gate requires ≥0.02 IoU. The U-Net clears it by +0.175 — **and it is still not
the default on real Indian scenes.** The two detectors disagree 2.5× on the real Barpeta
scene, traced to measured domain shift (training VV std 3.4 dB vs 6.2 dB here), so the
system serves the physically-grounded baseline until that is settled against ground
truth. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) §0.

### ML vs engineering — the honest split

| Component | Type | Why |
|---|---|---|
| SAR flood segmentation | **Real ML** — U-Net on Sen1Floods11 | Genuine CV problem, public labelled data |
| HAND depth estimation | Deterministic hydrology | Established method (Nobre et al. 2011). Not ML, and we say so |
| Road passability | Rules + spatial join | A lookup table is *better* here than a model |
| **Shelter assignment** | **Optimisation — min-cost flow** | The smartest part. Operations research, not ML |
| Routing | Dijkstra on a pruned graph | A graph algorithm is the correct tool |
| Multilingual cards | Pre-translated templates | Deterministic. A safety message must not be paraphrased |

One serious model, one exact optimiser, and a lot of careful engineering. **We do not add
models to look impressive.** A panel with a real ML person on it will respect
"we used a rule here because a model would be worse" far more than a network predicting
something a lookup handles.

---

## Honesty is built into the product

This is a design principle, not a disclaimer section:

- **The dashboard cannot show a clean screen over synthetic data.** The banner is generated
  from the scenario's own provenance, not from a flag someone might forget to set.
- **Estimated shelter capacity renders differently from surveyed capacity**, everywhere it appears.
- **Unobserved area is never reported as dry.** A Sentinel-1 pass covers ~69% of Barpeta;
  the remaining 31% is reported as *unknown*, and the flood extent as a *lower bound*.
- **Permanent water is subtracted** using JRC GSW, so we report flooding rather than the Brahmaputra.
- **CAP `status` cannot be set to `Actual`** without an explicit environment override.
  A prototype must not emit a live-looking public emergency alert by accident.
- **The U-Net only ships if it beats the Otsu baseline** on a held-out set, and both
  numbers go on the slide. If it loses, we ship the baseline and say so.

---

## Repository layout

```
config/          district YAML — every threshold lives here, each one cited
src/prahari/
  grid.py        metric raster grid; one spatial frame for every layer
  scenario.py    the object that flows through the pipeline, carrying provenance
  hazard/        HAND hydrology · Otsu baseline · U-Net · depth · metrics
  impact/        road passability · exposure · shelter viability
  decision/      pruning · min-cost flow · routing · priority ranking   ★
  delivery/      dashboard payloads · action cards · CAP 1.2 · i18n
  ingest/        real feeds, every one optional and degrading gracefully
  validate/      retrospective validation · permanent-water separation
  demo/          synthetic district · replay harness
  api/           FastAPI app
web/             zero-build MapLibre dashboard (works with no internet)
tests/           85 tests, offline, ~17 s
docs/            architecture · references · judge Q&A · validation · runbook
scripts/         run_real.py
```

---

## Documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How it works, and why each decision was made |
| [REFERENCES.md](docs/REFERENCES.md) | **Where every threshold comes from.** Read before presenting |
| [JUDGE_QA.md](docs/JUDGE_QA.md) | The questions you will be asked, and the answers |
| [VALIDATION.md](docs/VALIDATION.md) | What is validated, what is not, and how to check |
| [DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md) | Demo day: what to run, in what order, and what to say |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | What this system cannot do. Know these before a judge finds them |

---

## Testing

```bash
python -m pytest tests/ -q
```

85 tests, fully offline, about 17 seconds. The suite deliberately includes regression
tests for the two worst bugs found during development — a detector that reported more
flooding at low water than at high, and a dashboard that blanked entirely when WebGL
failed. Both were invisible to unit tests and obvious the moment the whole chain ran.

---

## Status

Working prototype. Runs end to end on real observations for Barpeta, Assam.
**Not an operational system** — see [LIMITATIONS.md](docs/LIMITATIONS.md).

Licensed MIT. Built for Smart India Hackathon 2026.
