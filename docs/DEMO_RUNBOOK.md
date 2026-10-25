# Demo runbook

What to run, in what order, and what to say.

---

## The night before

```bash
# 1. Cache everything. After this, the venue network is irrelevant.
python -m prahari.cli ingest --district barpeta --scenes 3
python -m prahari.cli ingest --district darbhanga --scenes 1

# 2. Warm the scenario cache for both districts at demo resolution
python -m prahari.cli serve --downsample 2 &
curl -s "http://127.0.0.1:8000/api/state?district=barpeta&frame=14" > /dev/null
curl -s "http://127.0.0.1:8000/api/state?district=darbhanga&frame=14" > /dev/null

# 3. Prove the whole suite is green
python -m pytest tests/ -q

# 4. Have the real run ready in a second terminal, already executed once
python -m prahari.cli real --district barpeta --date 2024-07-16
```

**Then stop changing things.** The most common way strong teams lose is adding one more
feature the night before and breaking the demo. Extra time removes the excuse for a
broken demo; it does not remove the temptation.

---

## Pre-flight, ten minutes before

- [ ] Server running, `http://127.0.0.1:8000` open and **already loaded once**
- [ ] Second terminal with the real-data run scrolled to its headline output
- [ ] Third terminal ready for `pytest`
- [ ] Wifi **off** — prove it works offline. Turn it on only for the basemap toggle
- [ ] Browser zoom set so the rail is readable from the back of the room
- [ ] `docs/JUDGE_QA.md` open on a phone, not on the demo screen

---

## The five-minute run

### 0:00 — The gap (30 s, no screen)

> India can already predict floods. IMD forecasts. CWC gauges rivers. Google Flood Hub
> does world-class AI riverine forecasting. SACHET has pushed 44 billion alerts.
>
> None of them can tell a family in Barpeta which of the four roads out of their village
> is still passable — or whether the school they're walking to already has 900 people in
> a 200-person hall.
>
> Every layer of forecasting exists. The decision layer doesn't exist at all. That's what
> we built.

### 0:30 — The map (60 s)

Open the dashboard. It is already loaded — do not load it live.

Point at the banner first:

> First thing: this banner is generated from the data's own provenance, not from a
> checkbox. We can't show you synthetic data pretending to be real even if we wanted to.

Then the map:

> Blue is water depth — and the colour changes exactly where the *advice* changes, not on
> some gradient. Pale is driveable. This shade is where a car loses control. Dark is where
> it's dangerous on foot.
>
> Red roads are cut. Click one.

*Click a blocked road.* Read the popup out loud: name, depth, status.

### 1:30 — Scrub the event (45 s)

*Drag the timeline.*

> This is a monsoon hydrograph — rises fast, recedes slowly. Watch the roads go red and
> the shelters fill.
>
> That's recomputing the entire decision chain per frame, in about a second. Terrain and
> hydrology are cached; only the flood state changes.

Stop at peak.

### 2:15 — The differentiator (90 s) ← **the important part**

Point at the boat dispatch list.

> This list is the thing no existing system produces. These settlements have **no road
> route** to any viable shelter. Not "a long way" — none. They're islands.
>
> Notice they're mostly *char* villages — the sandbar settlements on the Brahmaputra.
> That's not a coincidence in our model, it's the actual pattern in Barpeta.

Click one → the action card opens. Switch to Assamese.

> One screen. One destination. One route. And critically — *not* the main road, it's
> under water.

Then the counterfactual:

> We compute every route twice: once on the flooded network, once the way an ordinary
> navigation app would. **Google Maps will route you straight into the water. We route you
> around it — and we make sure the shelter we send you to still has room.**

Then capacity:

> Which is the second half. Nearest-shelter routing overflows the nearest shelter while
> one two kilometres away sits empty. We solve it as a minimum-cost flow — the capacity
> constraint is structural, the solver *cannot* overfill a shelter.
>
> And when there isn't enough capacity at all, we don't crash — we report the number. That
> number is the ask a district officer takes to the state.

### 3:45 — It's real (45 s)

Switch to the second terminal.

> Everything you've seen so far is a labelled simulation. Here's the same pipeline on the
> real Sentinel-1 scene from 16 July 2024, over the actual Barpeta flood. Real Copernicus
> DEM, real OpenStreetMap, real satellite.
>
> No API key. No waitlist. Free data, today.

Point at the coverage line:

> And note — that pass covered 69% of the district. We report the rest as *unobserved*,
> not dry. Under-reporting flooding where you have no information is the most dangerous
> bug this system could have.

### 4:30 — Scale and close (30 s)

*Switch the district dropdown to Darbhanga.*

> Nothing is hardcoded to Assam. One YAML file moves it to the Kosi-Bagmati in Bihar.
>
> One district, working end to end, on a pipeline built to generalise.

---

## Optional beats, if they ask

| Ask | Show |
|---|---|
| "Is your CAP real?" | `/api/cap/validate` — live structural validation |
| "How does it reach a feature phone?" | `/api/cap.xml` — SACHET's own schema, drives cell broadcast |
| "How do you know the code works?" | `python -m pytest tests/ -q` → 81 passed in ~17 s |
| "Can I have the dispatch list?" | `/api/priorities.csv` — control rooms run on paper |
| "What about the model?" | Engine chip, top-right: says exactly which detector is live |

---

## If something breaks

Say what happened and keep moving. **Do not narrate a fake recovery.**

> That's the live feed failing — which is exactly why replay mode exists. Here's July 2024.

If the map itself fails, the dashboard still works: every number, the dispatch list and
the action cards render without WebGL, by design. Point that out — it is a feature:

> The map's gone but the decision output hasn't. That's deliberate — a control room
> laptop that can't do WebGL still needs the dispatch list.

---

## Things not to say

- ❌ "It's 95% accurate" — quote IoU against a named held-out set, or say it's unvalidated
- ❌ "It works everywhere in India" — it works where there's a DEM, OSM roads and SAR; say that
- ❌ "This will save X lives" — you cannot support that number
- ❌ Anything about a component you cannot derive on a whiteboard

---

## The one-liner to leave them with

> India can already predict floods. It still can't tell a family which road is safe.
> That's the gap. That's all we built.
