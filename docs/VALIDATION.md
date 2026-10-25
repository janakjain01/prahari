# Validation

**This is the highest-value work in the project.** Almost no student team validates,
because it costs wall-clock time rather than cleverness — which is exactly why it
differentiates.

The sentence to be able to say:

> Run against the July 2024 event, PRAHARI flagged N villages as cut off. The actual
> relief operation reached those villages on day three.

---

## Results — actually run, July 2024 Barpeta

This is not a description of what the harness *would* produce. It is the output of

```bash
python -m prahari.cli validate --district barpeta --date 2024-07-16 --window 24 --frames 4
```

against four real Sentinel-1 acquisitions:

| Date | Swath coverage | Flood km² | River km² | Exposed | Cut off |
|---|---|---|---|---|---|
| 2024-06-22 | 69% | 338.3 | 12.0 | 251,534 | 5 (9,061) |
| **2024-06-24** | **100%** | **498.4** ← peak | 14.2 | 268,245 | 8 (16,830) |
| 2024-06-29 | 100% | 415.4 | 13.6 | 234,920 | **9 (15,628)** |
| 2024-07-04 | 69% | 376.8 | 15.8 | 270,020 | 5 (12,824) |

**Findings the harness reported:**

- **The series behaves like a real hydrograph.** Flood extent rises to a 498 km² peak on
  24 June and recedes afterwards. A detector returning noise over a real event would not
  produce that shape — and this check needs no external label, which matters because
  labels for Assam are not freely available.
- **Permanent water averaged 14 km²** of detected water and was subtracted. Without that
  step the system would be reporting the Brahmaputra itself as flooding.
- **At peak, 9 settlements (15,628 people) had no road route to any viable shelter.**
  That is the output no existing system produces, and it is the number to take to a
  district officer for confirmation against what actually happened.
- Swath coverage ranged 69–100%. Frames at different coverage are **not** directly
  comparable, and every extent is a lower bound.

Full report: `data/out/validation_barpeta_2024-07-16.{json,md}`

---

## Running it

```bash
python -m prahari.cli validate --district barpeta --date 2024-07-16
```

With an independent reference mask (a Bhuvan/NDEM export, or a hand-digitised extent):

```bash
python -m prahari.cli validate --district barpeta --date 2024-07-16 \
    --reference data/reference/bhuvan_20240716.tif
```

Writes `validation_barpeta_2024-07-16.json` and `.md`.

---

## What it checks

### 1. Permanent water is separated from flood

A SAR water mask over Barpeta **contains the Brahmaputra**. The Brahmaputra is always
there. Reporting it as flood inflates every number and makes the system look like it is
detecting a catastrophe on a dry day.

JRC Global Surface Water (Pekel et al. 2016) maps how often each 30 m pixel was water
between 1984 and 2021. Pixels above 80% occurrence are permanent channel and are
subtracted.

This is a **correctness fix** first and a validation input second. It is also the answer
to "aren't you just detecting the river?", which is among the first questions a
remote-sensing person asks.

The threshold is high deliberately: on a braided river the channel migrates, so a moderate
threshold would mark seasonally-wet sandbar as permanent and erase real flooding over the
chars — precisely where the vulnerable people are.

### 2. The flood behaves like a flood

The harness runs every Sentinel-1 acquisition in a window around the event and checks the
extent series **rises to a peak and recedes**.

This needs no external label, which matters because labels for Assam are not freely
available. A detector returning noise across a real hydrograph fails this test, and that
is exactly the class of bug that survives single-frame inspection — our tiled-Otsu bug
reported *more* flooding at low water than at high, and looked entirely plausible on any
one frame.

### 3. Extent against an independent reference — *if you supply one*

With `--reference`, flood extent is scored as IoU and F1 against an independent mask.

Without it, extent is checked for **internal and temporal consistency only**, and the
report says so in the "what this does not establish" section. We do not describe
consistency as accuracy.

### 4. Coverage is tracked per frame

A single Sentinel-1 pass covers ~69% of Barpeta. Every frame records its coverage, every
extent is reported as a **lower bound**, and the report notes explicitly that frames with
different coverage are not directly comparable to each other.

---

## What it does NOT establish

State these before a judge finds them.

| Not validated | Why |
|---|---|
| **Depth** | No public per-pixel depth truth exists for these events. HAND-derived depth is modelled throughout, and the road thresholds are depth thresholds — so this propagates. |
| **Shelter capacity** | Estimated from OSM footprints, never surveyed. The sheltered/unsheltered split inherits the error directly. |
| **The evacuation plan itself** | Nobody recorded which road each family took. This is the big one. |
| **Extent, without `--reference`** | Consistency is not accuracy. |

---

## The next step, and it is not code

The strongest available comparison is against **ASDMA relief-operation records**: which
villages were actually cut off, which camps actually filled, when boats reached whom.

Obtaining those is a conversation with a district officer, not an API call — which is why
it should be started early. One conversation with a District Disaster Management Authority
officer, an NDRF responder, or an ASDMA official will reshape this product more than a
week of coding.

Two questions worth asking them:

1. What do you actually do in the 48 hours before a flood?
2. What information do you wish you had?

A direct quote from a real official in the deck is unanswerable, and problem understanding
is the heaviest scoring criterion in SIH.

**This is pure wall-clock time to arrange. Start reaching out early.**

---

## Model validation (separate)

The hazard model has its own comparison, run inside training:

```bash
python -m prahari.hazard.train --data /path/to/sen1floods11 --epochs 40
```

This writes `models/training_report.json` containing:

- U-Net metrics on the **held-out test split**;
- Otsu baseline metrics on **the same tiles, in the same run**;
- a ship/no-ship verdict.

The U-Net only ships if it beats the baseline by **≥ 0.02 IoU**. Below that, the gain is a
different random seed, not an improvement — and we serve the baseline and say so.

A team that can show a model losing to a baseline, and choosing the baseline, reads as
more competent than one that shows only a winning number.

The training script also **refuses to mark a checkpoint shippable if it trained on
synthetic data**. The most embarrassing possible outcome is quoting a validation figure
that came from data you generated yourself.

---

## Reproducing the whole thing

```bash
python -m pytest tests/ -q                                   # 85 tests, offline, ~17 s
python -m prahari.cli demo --district barpeta --stage 3.0    # synthetic, no network
python -m prahari.cli real --district barpeta --date 2024-07-16   # real observations
python -m prahari.cli validate --district barpeta --date 2024-07-16
```

Every number in the pitch should be traceable to one of these four commands.
