# Setting up on a new machine

The repository holds **code only** — about 1.7 MB. Everything else is re-fetchable and
deliberately excluded:

| Excluded | Size | How to get it back |
|---|---|---|
| `data/raw/worldpop/` | 488 MB | `python -m prahari.cli fetch-population` |
| `data/raw/sen1floods11/` | 732 MB | `python -m prahari.cli fetch-training-data` |
| `data/cache/` | varies | rebuilt automatically on first run |
| `models/*.pt` | 95 MB each | `python -m prahari.cli train --epochs 40` |
| `cache/` | 28 MB | osmnx rebuilds it automatically |

Committing 1.3 GB of re-downloadable public data would make the repo unusable to clone.
None of it requires an account or an API key.

---

## 1. Install

```bash
git clone https://github.com/harsh01jain/sih_janak.git
```

```bash
cd sih_janak && pip install -r requirements.txt
```

**Windows + Anaconda note.** Anaconda and PyTorch each ship their own OpenMP runtime, and
importing both aborts with `OMP: Error #15`. The package sets `KMP_DUPLICATE_LIB_OK=TRUE`
on import to work around it, so nothing extra is needed — but if you see that error from
a script that imports `torch` *before* `prahari`, import `prahari` first.

**torchvision.** The pretrained ResNet-34 encoder needs a torchvision build matching your
torch. `pip install torchvision` may install an incompatible one and fail with
`operator torchvision::nms does not exist`. On torch 2.10, `torchvision==0.25.0` works:

```bash
pip install --no-deps torchvision==0.25.0
```

Without torchvision the model falls back to a non-pretrained encoder, which still trains
but scores lower.

---

## 2. Verify it works — no network needed

```bash
python -m pytest tests/ -q
```

85 tests, fully offline, about 15 seconds. If these pass, the decision core is sound.

```bash
python -m prahari.cli demo --district barpeta --stage 3.0 --downsample 4
```

Full synthetic run: flood detection, shelter assignment, boat dispatch, CAP export. No
downloads, no keys.

---

## 3. Bring in real data

Order matters only in that the first run of anything caches what later runs reuse.

```bash
python -m prahari.cli fetch-population
```

*(~488 MB, one time. WorldPop's server stalls on HTTP range requests, so the whole country
raster is downloaded once and read locally thereafter.)*

```bash
python -m prahari.cli ingest --district barpeta --scenes 3
```

*(Caches Copernicus DEM, OpenStreetMap, and Sentinel-1 scenes. Run this the night before
any demo — afterwards the venue network is irrelevant.)*

```bash
python -m prahari.cli real --district barpeta --date 2024-07-16
```

Full pipeline on the real July 2024 Assam flood.

---

## 4. Retrain the hazard model (optional)

```bash
python -m prahari.cli fetch-training-data
```

*(~732 MB, Sen1Floods11 from its public Google Cloud bucket. No account, no key.)*

```bash
python -m prahari.cli train --epochs 40 --batch-size 8
```

About 50 s/epoch on CPU, so roughly 35 minutes for 40 epochs. Writes
`models/unet_sar_flood.pt` and `models/training_report.json`, which contains the
model-vs-baseline comparison that decides whether the model ships.

Expected: test IoU ≈ 0.67 against an Otsu baseline of ≈ 0.50.

**Note the system does not use the U-Net by default on real Indian scenes** — see
`docs/LIMITATIONS.md` §0 for why. The dashboard runs fine with no checkpoint at all.

---

## 5. Run the dashboard

```bash
python -m prahari.cli serve --downsample 2
```

Open <http://127.0.0.1:8000>. Use the **Data source** toggle for replay or real satellite.

`--downsample 3` is faster to build; `1` is full 30 m resolution and slow on first run.

---

## Where to read next

| Document | For |
|---|---|
| `README.md` | What this is and why |
| `docs/PROJECT_SUMMARY.md` | Everything built, and what is verified |
| `docs/DEMO_RUNBOOK.md` | Demo day: what to run and what to say |
| `docs/JUDGE_QA.md` | The questions you will be asked |
| `docs/LIMITATIONS.md` | What it cannot do — read before presenting |
