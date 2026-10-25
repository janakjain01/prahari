# The questions you will be asked

Memorise the first five. The rest are the ones that catch teams out.

**One rule above all: nobody presents a component they cannot derive on a whiteboard.**
If the answer to "why did you model it that way?" is "that is how it was written", you
lose the room instantly — and it is a very common failure now. Whoever owns the
optimisation must be able to explain why greedy assignment overflows shelters, without
notes.

---

## The five you will definitely get

### 1. "How is this different from what IMD / CWC / Google already do?"

> Every one of those forecasts *where the water will be*. None of them tells a family
> *which road to take*. SACHET delivers alerts — it's a pipe, and a district-level
> "heavy rainfall likely" doesn't tell a family in Barpeta which of the four roads out
> of their village is still passable, or whether the school they're walking to already
> has 900 people in a 200-person hall.
>
> C-FLOOD does real hydrodynamic modelling, but only for Mahanadi, Godavari and Tapi.
> The Brahmaputra and the Kosi — India's worst flood zones — aren't covered.
>
> We consume their forecasts rather than replacing them, and we publish back in
> SACHET's own CAP 1.2 format. We're a decision layer inside the national pipeline,
> not beside it.

### 2. "Google Flood Hub already does this."

> Google does riverine forecasting better than we could, so we use their API as an
> input. Their published API surface has no routing and no shelter data. We're building
> on top of them, not against them.
>
> And we don't depend on them — their key is waitlisted, so the system is designed to
> run entirely on Sentinel-1 plus rainfall. That's the designed-for case, not a
> degraded one. If Google turned the API off tomorrow, nothing here stops working.

### 3. "Why hasn't the government done this?"

> C-FLOOD covers three basins after years of work because 2-D hydrodynamic modelling is
> expensive to set up per basin — you need bathymetry, calibration, boundary
> conditions. We take a lighter, generalisable approach: HAND normalises terrain against
> the drainage network, which trades some physical precision for national coverage.
>
> That's the right trade for an evacuation decision, where you need "which road, which
> shelter", not millimetre water depth. Swapping one YAML file moves us from Barpeta to
> Darbhanga — we can show that running.

### 4. "How accurate is your flood model?"

> Two numbers, and we show both. The classical Otsu baseline and the U-Net, on the same
> held-out set, from the same run. **The U-Net only ships if it beats the baseline by at
> least two IoU points** — otherwise we serve the baseline and say so on the slide.
>
> We built the baseline *first*, deliberately, so the ML has to earn its place instead
> of being decoration.

*If no checkpoint is trained yet, say exactly that:* "We're serving the classical
baseline right now — the dashboard says so in the corner. The training pipeline is
built and the Sen1Floods11 comparison harness runs in the same script."

### 5. "Where does shelter capacity data come from?"

> It's estimated, and the dashboard says so on every shelter. Surveyed capacity isn't
> open data. We take the OSM building footprint, apply the Sphere Handbook minimum of
> 3.5 m² per person, and discount to 65% for corridors and walls.
>
> Production needs the district relief-camp register from ASDMA. That's a real
> deployment dependency and we'd rather name it than invent numbers.

---

## The ones that catch teams out

### 6. "How does this reach someone without a smartphone?"

> CAP 1.2 export into SACHET's cell broadcast, which reaches feature phones with no app
> and no internet. That's why we emit CAP rather than inventing our own format — the
> answer already exists in the architecture. `/api/cap.xml` is live; we can validate it
> in front of you at `/api/cap/validate`.

### 7. "Can this scale beyond one district?"

> Nothing is hardcoded to Assam. `config/darbhanga.yaml` runs the same pipeline on the
> Kosi-Bagmati. Let me switch it in the dropdown.

*Show it. Don't assert it.*

### 8. "Why min-cost flow instead of just sending everyone to the nearest shelter?"

**This is the one to be able to derive on a whiteboard.**

> Because nearest-shelter isn't a decision, it's a lookup — and it breaks the moment two
> villages share a nearest shelter.
>
> Village A has 1,200 people, village B has 900, both closest to the same school, which
> holds 800. Greedy sends 2,100 people to a building that holds 800. The other 1,300
> find out *after* walking there, in the dark, in the rain, with a river rising. And a
> second school 2 km further on stands empty.
>
> The right question isn't "what's nearest?" It's "what allocation minimises total
> travel time *subject to* no shelter exceeding capacity?" That's a transportation
> problem, and min-cost flow solves it exactly — not heuristically.
>
> The capacity constraint is structural: flow can only leave a shelter node through an
> arc capped at its capacity, so the solver *cannot* overfill it. It's not a check we
> run afterwards.

**Follow-up you should expect — "what if there isn't enough capacity at all?"**

> Then it's infeasible, and a program that crashes in exactly the situation it was built
> for is useless. So every settlement also has an arc to an `UNSHELTERED` node at a
> penalty far above any real travel cost. That makes the objective lexicographic: first
> maximise how many people get sheltered, then minimise travel time among those
> solutions. And "infeasible" becomes a *number* — how many people we cannot shelter.
> In Assam that number is the most operationally useful thing we produce. It's the ask a
> district officer takes to the state.

### 8b. "Your U-Net and your baseline disagree — which one is right?"

**Answer this eagerly. It is the strongest honesty signal you have.**

> We don't know yet, and we'd rather tell you that than pick one.
>
> On Sen1Floods11's held-out split the U-Net wins clearly — IoU 0.65 against the
> baseline's 0.50. But on the real Barpeta scene they disagree by 2.5×: Otsu says
> 320 km², the U-Net says 131 km².
>
> We measured why. The training data has VV standard deviation of 3.4 dB; this scene has
> 6.2 dB, with much brighter extremes. Sen1Floods11 is 11 global events processed as GRD;
> we're inferring on RTC over a braided Himalayan floodplain. That's domain shift, and the
> U-Net has no terrain prior to fall back on.
>
> So we ship the baseline by default on real Indian scenes. It's physically grounded,
> terrain-vetoed, explainable on a whiteboard, and it errs toward *more* water — which is
> the right direction to be wrong in during an evacuation.
>
> To settle it properly we need a reference mask. `--reference` already takes a
> Bhuvan/NDEM export and scores IoU against it. Until then we don't quote either
> extent as fact.

*If asked "so was the ML worth it?":*

> On the benchmark it's a genuine +0.16 IoU, and the pipeline is ready the moment we have
> Indian ground truth to fine-tune on. But we're not going to serve a model on a district
> where we can't yet show it's better — that's exactly the ship/no-ship gate we built.

### 9. "Aren't you just detecting the river?"

> No — and that's a real failure mode, so we handle it explicitly. We subtract JRC Global
> Surface Water, which maps how often each pixel was water between 1984 and 2021.
> Anything above 80% occurrence is permanent channel and comes out. The dashboard reports
> flood and river separately.
>
> We use a high threshold deliberately: the Brahmaputra is braided and the channel
> migrates, so a moderate threshold would mark seasonal sandbar as permanent and erase
> flooding over the chars — which is exactly where the vulnerable people live.

### 10. "Your satellite doesn't cover the whole district."

> Correct, and we report that rather than hiding it. A single Sentinel-1 pass covers
> about 69% of Barpeta. The other 31% is marked **unobserved, not dry** — the flood
> extent we report is explicitly a lower bound, and the API returns a coverage caveat.
>
> Reporting unobserved area as dry would be the most dangerous bug this system could
> have, because it under-reports flooding precisely where we have no information.

### 11. "Why SAR and not optical imagery?"

> Floods happen during the monsoon. During the monsoon India is under complete cloud
> cover, so optical satellites see nothing. SAR penetrates cloud. It's the only imagery
> that works on the day it matters.

### 12. "What did you use AI/ML for, honestly?"

> One serious model — U-Net segmentation on Sen1Floods11 — and one exact optimiser for
> shelter assignment. Everything else is deterministic hydrology, rules, and graph
> algorithms.
>
> Depth comes from HAND, not a neural network, because a neural network would be worse
> there. Road passability is a lookup table because it *is* a lookup table. We didn't
> add models to look impressive.

### 13. "How long does it take to run?"

> Static layers — terrain, HAND, road graph — are computed once per district and cached.
> After that, a full flood-state recompute is about a second at district scale, which is
> why the replay scrubber is instant. Let me drag it.

### 14. "Is this data real, or did you generate it?"

> Both, and the dashboard always tells you which. The banner is generated from the
> scenario's own provenance record, not from a flag we set by hand — so it's physically
> impossible for us to present synthetic data as real.
>
> Right now you're looking at a [synthetic / replayed / live] scenario. Here's the same
> pipeline on the real 16 July 2024 Sentinel-1 scene.

*Have the real run ready in a second terminal.*

### 15. "What are the limitations?"

Answer this one *eagerly* — see [LIMITATIONS.md](LIMITATIONS.md). A team that names its
own weaknesses before the panel finds them reads as competent. A team that gets caught
hiding one loses everything else it said.

Lead with the sharpest one:

> The biggest one is that a 30 m DEM cannot resolve embankments. An Assam embankment is
> a few metres wide — sub-pixel. So we can't see an intact embankment holding water back
> or a breach opening, and breached embankments are a primary flood mechanism in this
> district. We partially mitigate by burning OSM dyke centrelines into the DEM, but OSM's
> coverage of flood infrastructure here is sparse. Fixing it properly needs the Water
> Resources Department's embankment inventory.

---

## If the demo breaks

Say what happened, plainly, and keep going. Do not narrate a fake recovery.

> That's the live feed failing — this is exactly why we built replay mode. Here's the
> July 2024 event.

Judges have seen many demos break. What they punish is pretending.

## If you don't know

> I don't know. Here's how I'd find out.

That answer costs you almost nothing. Bluffing and getting caught costs you the room.
