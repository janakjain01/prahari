"""Shared contract for every ingestion adapter.

One rule, applied everywhere: an adapter NEVER raises for an expected failure.
Network down, key missing, dependency not installed, nothing in the archive
for that date — all of those return ``IngestResult(ok=False, reason=...)``.

The reason is that ingestion runs in three very different contexts and the
right response differs in each:

    the CLI            wants to print the problem and carry on with the rest
    the API            wants to degrade to cached or synthetic data
    the validation run wants to fail loudly

A raise takes that choice away from the caller. Returning a value leaves it
where it belongs, and it means one flaky feed cannot take down a demo.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

__all__ = ["IngestResult", "requires", "cache_path", "retry"]

T = TypeVar("T")


@dataclass
class IngestResult(Generic[T]):
    """Outcome of one fetch. Truthy when it worked."""

    ok: bool
    data: Optional[T] = None
    source: str = ""
    detail: str = ""
    reason: str = ""
    from_cache: bool = False
    elapsed_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    def unwrap(self) -> T:
        """Return the data, raising if the fetch failed.

        For callers that genuinely cannot continue — the validation harness,
        mostly. Ordinary callers should check ``ok`` and degrade.
        """
        if not self.ok or self.data is None:
            raise RuntimeError(f"{self.source or 'ingest'} failed: {self.reason}")
        return self.data

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "source": self.source,
            "detail": self.detail,
            "reason": self.reason,
            "from_cache": self.from_cache,
            "elapsed_s": round(self.elapsed_s, 2),
            **({"meta": self.meta} if self.meta else {}),
        }

    @classmethod
    def failed(cls, source: str, reason: str) -> IngestResult[T]:
        logger.warning("%s unavailable: %s", source, reason)
        return cls(ok=False, source=source, reason=reason)


def requires(*modules: str) -> Callable:
    """Decorator: turn a missing optional dependency into a clean failure.

    Without this, an absent ``rasterio`` raises ImportError from deep inside a
    call stack during a demo. With it, the caller gets "rasterio is not
    installed" and falls back.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import importlib

            missing = []
            for module in modules:
                try:
                    importlib.import_module(module)
                except ImportError:
                    missing.append(module)
            if missing:
                return IngestResult.failed(
                    func.__name__,
                    f"missing optional dependency: {', '.join(missing)}. "
                    f"Install with: pip install 'prahari[geo,remote]'",
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def retry(attempts: int = 3, backoff_s: float = 1.5) -> Callable:
    """Retry transient network failures with exponential backoff.

    Remote archives return 503 under load often enough that a single attempt
    is not a fair test of availability.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - deliberately broad
                    last = exc
                    if attempt < attempts - 1:
                        delay = backoff_s * (2 ** attempt)
                        logger.info(
                            "%s attempt %d/%d failed (%s); retrying in %.1fs",
                            func.__name__, attempt + 1, attempts, exc, delay,
                        )
                        time.sleep(delay)
            return IngestResult.failed(func.__name__, f"{type(last).__name__}: {last}")

        return wrapper

    return decorator


def cache_path(district_id: str, kind: str, key: str, suffix: str = ".npz") -> Path:
    """Deterministic on-disk cache location.

    CACHE EVERYTHING. On demo day the venue network will be hostile or absent,
    and every response we fetched during preparation must still be on the
    laptop. This is listed in the risk register as "cache every API response
    locally; never depend on venue wifi", and this function is how that is
    actually enforced rather than remembered.
    """
    from ..config import data_dir

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    path = data_dir() / "cache" / district_id / kind / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()
