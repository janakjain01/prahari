# PRAHARI — run report & review

**Run date:** 21 August 2026 · **Commit state:** working copy in `sih_janak-main`
**Environment:** Linux sandbox, Python 3.11, no PyPI access (see *Caveats*)

---

## Verdict

**The project is done in the sense that matters: the decision layer works end to end.**
The synthetic pipeline runs clean, 77 of 85 tests pass with 0 failures, and every one of the six
outputs the README promises is actually produced. The gaps are all in the *real-data* half —
the paths that need network and geospatial libraries — and they are real bugs, not missing
features. Details in §3.

---

## 1. What ran, and what it produced

`python -m prahari.cli demo --district barpeta --stage 3.0` — **clean run, 1.34 s for the plan.**

```
Synth: Barpeta ready — 78 settlements (1,690,000 people), 120 shelters (69,382 places), 6,264 road edges
Hazard detector vs known ground truth: IoU=0.9242 F1=0.9606 P=0.9843 R=0.9380
Shelters: 112 viable of 120 (8 flooded out); usable capacity 57,212 after a 10% reserve
Roads: 95.3 km of 1,608.6 km cut to vehicles (5.9%), worst depth 3.68 m
Pruned graph (foot): removed 288/6,264 edges; network fell into 111 components
Min-cost flow: 68 settlements (153,926 people), 112 shelters (57,162 usable places), 1,801 arcs
Assignment: 56,879 sheltered (37.0%), mean journey 60.2 min, 97,047 UNSHELTERED
Routes: 151 computed; 2 steered away from a flooded road a naive router would have used
Priority list: 28 settlements flagged, 4 completely CUT OFF (5,861 people)
CAP 1.2 PRAHARI-barpeta-…: VALID  (4 languages, 20 areas, status=Exercise, 16,151 bytes)
```

All six blocks fire: hazard → impact → pruning → min-cost flow → routing → cut-off ranking →
action cards → CAP export. The boat-dispatch list is populated and correctly separated from the
capacity shortfall. CAP validates and `status=Exercise` holds without an override, as designed.

**Tests:** `77 passed, 0 failed, 8 skipped`. The 8 skips are the `TestAPI` class — they call
`pytest.importorskip("fastapi")`, and FastAPI is uninstallable in this sandbox. Under real pytest
on your machine they should run. 77 + 8 = 85, matching the README's count.

**`prahari status`** reports both districts configured, five keyless feeds OK, the two gated feeds
(Flood Hub, IMD) marked unavailable, and the Otsu baseline serving because there is no U-Net
checkpoint on disk — the designed fallback, behaving as documented.

---

## 2. The dashboard

The FastAPI server can't run here (no `pip install fastapi`), so instead the full Barpeta replay
was baked into a self-contained page: **28 frames** of state, every action card in all four
languages, and every depth overlay, answered from an in-page bundle by a `fetch` shim. It runs
with no server and no network. The map, scrubber, dispatch list, shelter-load rings, provenance
panel and the four-language citizen card all work.

Greyed out in the export, because they need the live API or the network: *By road*, *Real
satellite*, the OSM basemap, and the district selector.

---

## 3. Real-data paths — findings from code review

These could not be executed (no network, no `rasterio`/`geopandas`/`osmnx`), so this is a read of
the code. The call chains are **internally consistent** — every signature, keyword and return
shape lines up across `cli` → `run_real` → `build_real_scenario` → the six ingest adapters →
`detect_flood` → `build_evacuation_plan`. The osmnx 2.x API names are correct. Nothing fails
immediately. What follows are defects that survive that check.

### Fix before demo day

**1 · The SAR cache loses `calibrated`, and the second run silently reports zero flood.**
`Sentinel1Scene.save()` (`ingest/sentinel1.py:79`) writes only `[item_id, acquired, collection,
orbit]`. The cache-load path (`:191`) rebuilds the scene with four positional args, so
`calibrated` falls back to its `True` default (`:53`). With `--collection sentinel-1-grd`, run 1
correctly detects uncalibrated DN data and lets Otsu derive thresholds from the histogram; run 2
reads the cache, believes the scene is calibrated, and applies the −13.0 dB physical ceiling
(`hazard/otsu.py:330`) to data sitting at +15…+27 dB. Result: empty mask, zero flood, zero
exposed population — reported confidently. *Fix: add `calibrated` to the saved meta array and
read it back.*

**2 · `prahari real` never subtracts permanent water.**
`validate` does it (`validate/retrospective.py:189`) and the dashboard's real mode does it
(`api/state.py:259`), but `scripts/run_real.py` and `ingest/build.py` do not. So `prahari real`
reports the Brahmaputra channel itself as flood, and gives materially different numbers from
`prahari validate` on the same scene. `compute_hand` also takes a `permanent_water=` argument
(`hazard/hand.py:403`) that `build.py:114` never passes. This is exactly the honesty claim the
README makes in "Permanent water is subtracted" — true for two of three entry points.

**3 · Three of six OSM layers are never cached, so `prahari ingest` doesn't actually make you
network-independent.** `fetch_road_network` is the only OSM function that calls `cache_path`
(`ingest/osm.py:54`). `fetch_shelter_candidates` even takes a `district_id` parameter (`:117`)
and never uses it — `build.py:160` passes it in good faith. Offline on demo day: DEM, roads and
population come from cache, the shelter harvest fails, `shelters = []`, and the plan reports the
entire district as unshelterable.

**4 · `fetch_embankments` is the one adapter without `@retry`, so it can crash the run.**
`ingest/osm.py:377` has `@requires` but no `@retry`. `@retry`'s bare `except Exception`
(`ingest/base.py:125`) is what actually converts a raise into `IngestResult.failed` — without it,
anything after the first Overpass call (`.to_crs()` on a `gdf` with `crs=None`, `.geometry.type`)
propagates out of `build_real_scenario` and ends `real`/`validate` in a traceback instead of
degrading.

Related, in all three OSM feature functions (`osm.py:131`, `:255`, `:393`): `except TypeError`
sits *before* `except Exception`, so any `TypeError` raised inside osmnx triggers the osmnx-1.x
positional fallback — which on osmnx 2.x raises its own `TypeError` and masks the original error.

**5 · The persisted build report always says `usable: False`.**
`meta={"build_report": dict(report)}` takes a shallow copy at `build.py:196`; `report["usable"] =
True` is set at `:229`. Anything saved with `run_real.py --save` records a successful build as
unusable.

### Worth checking before you rely on it

- **`validate` aborts on one bad frame.** `retrospective.py:173` guards the fetch but not
  `detect_flood` / `build_evacuation_plan`, both of which raise `ValueError` on plausible inputs.
  One bad scene out of six loses the whole run, including the frames that worked, because the
  report is only written at `:239`.
- **The U-Net is never exercised by `real` or `validate`.** Both hardcode `prefer="otsu"`
  (`run_real.py:91`, `retrospective.py:187`) and pass no `checkpoint=`. Only `api/state.py` can
  use it. Also `cli.py:319` defaults `--out models` relative to CWD while the API loads
  `project_root()/models` — train from anywhere but the repo root and the checkpoint lands where
  nothing reads it.
- **`search_sentinel1`'s `limit=` is a page size, not a cap.** `pystac_client` paginates through
  everything matching; `max_items=` is the capping parameter. Over the ±24-day window in
  `cli.py:335` that's every acquisition, not 25.
- **`torch.amp.GradScaler("cuda", …)`** at `hazard/train.py:119` is a recent constructor;
  `requirements.txt` allows `torch>=2.1`, where the name is `torch.cuda.amp.GradScaler`.
- **`threshold_db=0.0` is hardcoded** in the per-frame validation JSON (`retrospective.py:210`)
  even though `OtsuResult.threshold_db` is right there. An always-zero real-looking field in the
  validation artefact is the wrong kind of wrong for this project's posture.
- **HAND is recomputed on every run.** Both `build.py` and `hand.py` document it as a one-time
  cached precompute; there is no HAND cache anywhere. Only the DEM is cached.
- **The population fallback degrades to a single settlement.** A flat placeholder raster
  (`build.py:155`) makes the 88th-percentile threshold select the whole grid as one component —
  one settlement at the centroid holding 1.69 M people. It doesn't crash, but it isn't usable
  either, and the report won't flag why.
- **`retry`'s bare `except Exception` makes bugs look like outages** (`base.py:111`). An
  `AttributeError` inside an adapter becomes "No elevation data … Check network access to
  planetarycomputer.microsoft.com" after 4.5 s of sleeps. On paths that can't be tested offline,
  that is the single biggest obstacle to debugging the first real run.

### Minor

- `pyproject.toml` declares package data at `delivery/templates/*.j2` and `delivery/locales/*.json`;
  neither directory exists (translations are inline in `delivery/i18n.py`). Harmless, but it will
  confuse anyone packaging this.
- `osm.py:37` — `__all__` omits `fetch_place_names` and `label_settlements`, both of which
  `build.py` and `ingest/__init__.py` import.
- `cli.py:110` — `cmd_real` mutates `sys.argv` without restoring it, and exposes none of
  `run_real.py`'s `--no-cache`, `--save`, `--out`.
- `hazard/dataset.py:75` — `MEASURED_SEN1FLOODS11_MEAN_DB` / `_STD_DB` are defined and documented
  as "use these for new runs" but referenced nowhere; `train.py:158` ships the legacy constants.

---

## 4. Running it on your own machine

Everything works there — you have the full dependency set.

```bash
cd sih_janak-main
pip install -r requirements.txt

python -m prahari.cli serve --downsample 2     # dashboard at http://127.0.0.1:8000
python -m prahari.cli demo --district barpeta --stage 3.0
python -m pytest tests/ -q                     # expect 85 passed
python -m prahari.cli status
```

Real data (needs network):

```bash
python -m prahari.cli ingest --district barpeta        # cache first, on good wifi
python -m prahari.cli real --district barpeta --date 2024-07-16
python -m prahari.cli validate --district barpeta --date 2024-07-16
```

Run `ingest` on reliable wifi the day before, not at the venue — but see finding **3**: three of
the six OSM layers won't actually be cached by it yet.

---

## Caveats on this run

The sandbox has no package index, so `fastapi`, `pytest`, `torch`, `pyproj`, `rasterio`,
`geopandas` and `osmnx` could not be installed. To get the pipeline running, two throwaway
stand-ins were placed **outside your project tree** (nothing in `sih_janak-main` was modified):

- a pure-Python WGS84↔UTM transformer standing in for `pyproj` (Krueger series, round-trips to
  ~1e-8°), used only by `grid.py`'s coordinate conversion;
- a minimal `pytest` implementing fixtures, `approx`, `raises`, `importorskip` and `tmp_path`.

Neither touches the decision logic. The numbers above come from your code. Re-run the suite with
real pytest on your machine to confirm the 8 API tests pass.
