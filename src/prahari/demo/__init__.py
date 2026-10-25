"""Demo harness — the system must never fail live on stage.

Two modes, and building both is not optional:

    LIVE    today's real feeds, running now
    REPLAY  a historical flood event, stepped forward hour by hour

On pitch day there will almost certainly be no active flood in Barpeta. Replay
mode is how the system is shown doing its job. Label it honestly:

    "This is a replay of the July 2024 event. The live feed is running in the
     other tab — here's today's actual rainfall coming in."

Judges respect an honest replay plus a live feed enormously more than a demo
that pretends. Teams get caught faking demos constantly; clearly labelling the
replay is a trust signal, not a weakness.

``synth.py`` additionally provides a fully synthetic district so that the
entire pipeline — including SAR water detection — is exercisable with zero
downloads and zero API keys. Everything it produces is stamped
``is_synthetic=True`` and the dashboard shows a banner. It is for development
and for the worst-case "no internet at the venue" scenario, never for a number
on a slide.
"""

from .synth import synthesise_district, synthesise_sar, SynthConfig
from .replay import ReplayEvent, ReplayHarness, build_replay_from_stages

__all__ = [
    "synthesise_district",
    "synthesise_sar",
    "SynthConfig",
    "ReplayEvent",
    "ReplayHarness",
    "build_replay_from_stages",
]
