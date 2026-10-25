"""Download Sen1Floods11 from its public Google Cloud Storage bucket.

Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020) is the labelled dataset the
hazard model trains on. The bucket is PUBLIC and anonymously readable over plain
HTTPS — no Google account, no `gcloud` install, no API key, no registration.
That matters: it means the one genuinely learned component of this system can be
trained without asking anyone for anything.

We fetch only the hand-labelled subset actually used for training:

    v1.1/data/flood_events/HandLabeled/S1Hand/     446 files, ~730 MB  (VV+VH)
    v1.1/data/flood_events/HandLabeled/LabelHand/  446 files, ~2.6 MB  (labels)
    v1.1/splits/flood_handlabeled/                 4 files            (splits)

The much larger weakly-labelled subset is deliberately skipped — the hand-labelled
split is what the published benchmarks use, and it is what makes our
model-vs-baseline comparison meaningful.

Downloads are resumable: a partial file is written to ``.part`` and only renamed
on success, so an interrupted run never leaves a truncated GeoTIFF that later
reads as corrupt data.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

__all__ = ["download_sen1floods11", "list_bucket", "PREFIXES"]

BUCKET = "sen1floods11"
API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
MEDIA = "https://storage.googleapis.com/{bucket}/{obj}"

PREFIXES = (
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "v1.1/splits/flood_handlabeled/",
)


def list_bucket(prefix: str, timeout: float = 90.0) -> list[dict]:
    """List every object under ``prefix``, following pagination."""
    items: list[dict] = []
    token: Optional[str] = None
    while True:
        url = API.format(bucket=BUCKET) + "?maxResults=1000&prefix=" + urllib.parse.quote(prefix)
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        request = urllib.request.Request(url, headers={"User-Agent": "PRAHARI/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items.extend(payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            break
    return items


def _download_object(name: str, target: Path, timeout: float = 180.0) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    url = MEDIA.format(bucket=BUCKET, obj=urllib.parse.quote(name))
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "PRAHARI/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        part.replace(target)
        return True
    except Exception as exc:
        logger.warning("failed %s: %s", name, exc)
        part.unlink(missing_ok=True)
        return False


def download_sen1floods11(
    root: str | Path,
    prefixes: tuple[str, ...] = PREFIXES,
    skip_existing: bool = True,
) -> dict:
    """Download the hand-labelled subset into ``root``.

    Returns a report dict. Safe to re-run: existing files of the right size are
    skipped, so an interrupted download resumes where it stopped.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0
    total_bytes = 0

    for prefix in prefixes:
        objects = list_bucket(prefix)
        logger.info("%s — %d objects", prefix, len(objects))
        for i, obj in enumerate(objects, start=1):
            name = obj["name"]
            size = int(obj.get("size", 0))
            target = root / name

            if skip_existing and target.exists() and target.stat().st_size == size:
                skipped += 1
                continue

            if _download_object(name, target):
                downloaded += 1
                total_bytes += size
            else:
                failed += 1

            if i % 25 == 0 or i == len(objects):
                logger.info(
                    "  %s %d/%d (%d new, %d cached, %d failed, %.0f MB)",
                    prefix.rsplit("/", 2)[-2], i, len(objects),
                    downloaded, skipped, failed, total_bytes / 1e6,
                )

    report = {
        "root": str(root),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "megabytes": round(total_bytes / 1e6, 1),
    }
    logger.info("Sen1Floods11: %s", report)
    if failed:
        logger.warning(
            "%d object(s) failed. Re-run the same command — completed files are skipped.",
            failed,
        )
    return report


def main() -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Download Sen1Floods11 (public, no auth)")
    parser.add_argument("--root", default="data/raw/sen1floods11")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    report = download_sen1floods11(args.root)
    print(json.dumps(report, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
