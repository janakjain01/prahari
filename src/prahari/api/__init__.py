"""HTTP API for the responder dashboard.

Kept import-light: pulling in FastAPI at package-import time would make the
CLI and the test suite depend on the web stack for no reason.
"""

__all__ = ["app", "get_engine", "Engine"]


def __getattr__(name: str):
    if name == "app":
        from .main import app

        return app
    if name in ("get_engine", "Engine"):
        from . import state

        return getattr(state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
