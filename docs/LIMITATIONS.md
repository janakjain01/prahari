# Limitations

**Know these before a judge finds them.** A team that names its own weaknesses reads as
competent; a team caught hiding one loses credibility for everything else it said.

Ordered by how much they matter.

---

## 0. The two detectors disagree on real Indian scenes, and we cannot yet say which is right

**The finding.** On the real Barpeta Sentinel-1 RTC scene of 16 July 2024:

| Detector | Flood detected | % of observed swath |
|---|---|---|
| Otsu baseline | 320.5 km² | 25.6% |
| U-Net | 130.7 km² | 7.2% |

A 2.5× disagreement. The U-Net wins clearly on Sen1Floods11's held-out split
(IoU 0.654 vs 0.499), so this is not simply "the model is worse".

**The likely cause — measured, not guessed.** The training data and this scene have
visibly different radiometry:

| | VV mean | VV std | VV p99 |
|---|---|---|---|
| Sen1Floods11 (measured) | −9.69 dB | **3.43** | −3.8 dB |
| Barpeta RTC scene | −10.95 dB | **6.21** | **+1.6 dB** |

The real scene has nearly double the spread and much brighter extremes. Sen1Floods11
covers 11 global flood events processed as GRD; we infer on RTC over a braided
Himalayan floodplain. That is textbook domain shift, and the U-Net has no terrain prior
of its own to fall back on.

**What we do about it.** The system **defaults to the Otsu baseline on real scenes**
(`detector=otsu`), because it is physically grounded, terrain-vetoed, explainable on a
whiteboard, and errs toward *more* water. In an evacuation a false "flooded" costs a
detour; a false "dry" costs lives. The U-Net remains available via `?detector=unet`.

We also now ship the normalisation statistics **inside the checkpoint**, so a
train/inference mismatch cannot silently degrade predictions the way it could before.

**How to settle it.** Score both against an independent reference mask —
`prahari validate --reference <bhuvan_export.tif>` already supports exactly this. Until
that is done, **neither flood-extent figure should be quoted as fact.**

This is a good thing to raise with judges rather than hide: "our two methods disagree,
here is the measured reason, here is which one we ship and why, and here is how we would
settle it" is a stronger answer than a single confident number.

---

## 1. A 30 m DEM cannot resolve embankments

**The problem.** An Assam embankment is a few metres wide and a couple of metres high —
sub-pixel in both dimensions on a 30 m DEM. So the terrain model cannot see an intact
embankment holding water back, and cannot see a breach opening. In Barpeta, breached
embankments are a *primary* flood mechanism, so this is a real constraint on the whole
hydrology.

**What we do.** `ingest/dem.py::burn_embankments()` raises DEM cells along OSM dyke and
embankment centrelines before HAND is computed. It is applied automatically and recorded
in provenance.

**Why that is only a partial fix.** OSM's coverage of flood infrastructure in rural Assam
is sparse — the real Barpeta fetch found **one** mapped embankment line for the entire
district. Proper treatment needs the Assam Water Resources Department embankment
inventory, which is not open data.

---

## 2. Shelter capacity is estimated, never surveyed

Every capacity figure comes from an OSM building footprint × Sphere's 3.5 m²/person ×
0.65 usable-floor discount. None is a surveyed number, because surveyed capacity is not
open data.

Consequence: the sheltered/unsheltered split inherits that uncertainty directly. If our
footprint estimate is 30% high, the reported shortfall is 30% low.

The dashboard renders estimated capacity differently from surveyed capacity everywhere
it appears, and the API returns an explicit caveat. Production deployment requires the
district relief-camp register from ASDMA.

---

## 3. OSM completeness is uneven, and it biases the priority list

Some Assam villages have every lane mapped; some have one line through the middle. The
real Barpeta fetch returned **14** shelter candidates and 3,696 road edges for a
district of 1.7 million people. That is an OSM coverage artefact, not a description of
Barpeta.

This affects the output directly: **a settlement can appear cut off because its road was
never mapped.** We do not merge that case into `CUT_OFF` — unbound settlements get their
own reason code (`UNBOUND`) with the action "verify access on the ground". But the
underlying ambiguity remains.

---

## 4. Depth is modelled, and it is not validated

HAND-derived depth assumes:

- **static water** — no momentum, no velocity field;
- **the DEM predates the flood**, so channel scour is invisible;
- **a locally flat water surface** within each connected flood body.

There is no public per-pixel depth ground truth for these events, so depth is
**unvalidated** throughout. Extent can be checked; depth cannot. Since the road
passability thresholds are depth thresholds, uncertainty in depth propagates into which
roads are called cut.

---

## 5. One satellite pass does not cover the district

A single Sentinel-1 acquisition covers roughly **69%** of Barpeta. The remainder is
unobserved.

We report this rather than hiding it: unobserved area is marked *unknown*, never *dry*,
and reported flood extent is explicitly a **lower bound**. Frames with different coverage
are not directly comparable to each other.

Mitigation is to mosaic adjacent-orbit acquisitions, at the cost of a temporal offset
between the two halves of the district — which introduces its own error during a rapidly
rising flood.

---

## 6. SAR systematically under-detects urban flooding

Water is radar-dark because it reflects energy away. But buildings **double-bounce**
radar back to the sensor, so a flooded street between buildings can appear bright and be
classified as dry. This is a well-known limitation of SAR flood mapping, not a bug in our
implementation, and it means flooding in the denser parts of Barpeta town is likely
under-reported.

The synthetic generator deliberately reproduces this effect so the demo does not look
better than reality.

---

## 7. The evacuation plan itself is unvalidated

We can check flood extent against imagery. We cannot check the *plan*, because nobody
recorded which road each family took during the 2024 flood.

The strongest available comparison is against ASDMA relief-operation records — which
villages were actually cut off, which camps actually filled. Obtaining those is a
conversation with a district officer, not an API call. It is the single highest-value
piece of remaining work.

---

## 8. Population is modelled, and it is not a live headcount

WorldPop disaggregates census totals onto a 100 m grid using built-up area. It is good
for "roughly how many people are in this square kilometre" and wrong for any statement
about a household.

More importantly, it is a *resident* count from a census year. During a monsoon flood,
seasonal labour movement and prior self-evacuation both mean the people actually present
differ from the raster — usually fewer, sometimes many fewer.

We aggregate reporting to 1 km cells so the dashboard never implies a precision the data
does not have.

---

## 9. Assignment is divisible; real evacuation is not

Min-cost flow splits a village of 1,200 as 800/400 across two shelters, which is what
actually happens. But it splits at the *settlement* level, not the household level — the
model has no notion of keeping families together, or of the fact that people follow their
neighbours regardless of what an optimiser says.

The plan is a recommendation to a district officer, not a prediction of behaviour.

---

## 10. Translations are not native-speaker validated

The Assamese, Bengali and Hindi alert strings were prepared by the team. They have **not**
been reviewed by a native Assamese speaker or by ASDMA.

Alert wording is a life-safety surface. "It looked right to us" is not a standard. The
API returns this caveat with every card, and review is a hard prerequisite for any real
deployment.

---

## 11. It is a prototype, and it says so

- CAP `status` is `Exercise`, and cannot be set to `Actual` without an explicit
  environment override. A prototype must not emit a live-looking public emergency alert.
- Nothing here is certified, accredited, or approved by NDMA, ASDMA or IMD.
- No part of this has been tested with people who would actually rely on it.

---

## What would change these

| Limitation | What it needs |
|---|---|
| Embankments | Assam WRD embankment inventory |
| Shelter capacity | ASDMA relief-camp register |
| OSM gaps | A local mapping effort, or state road inventory |
| Depth validation | Field depth marks, or a calibrated hydrodynamic run to compare against |
| Plan validation | ASDMA relief-operation records for 2024 |
| Translations | A native Assamese speaker, and ideally ASDMA sign-off |
| Urban under-detection | Optical or interferometric coherence fusion where cloud allows |

Most of these are **conversations, not code**. That is worth saying out loud: the
remaining work on this system is mostly institutional access, not engineering.
