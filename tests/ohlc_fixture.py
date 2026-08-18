"""The fixed OHLC fixture, shared by the charting and trader tests.

Built from an explicit price path, so every pivot is there by construction and the
golden assertions in `test_charting.py` can be checked by hand. No clock, no
randomness — the frame is byte-identical on every run and every machine.
"""

from __future__ import annotations

from typing import Sequence

from src.agents.charting import Bar

EPOCH = 1_755_000_000  # 2025-08-12 12:00:00Z — fixed, so the session window is fixed
H1 = 3600
WICK = 0.0002


def leg(start: float, end: float, steps: int) -> list[float]:
    return [start + (end - start) * (i + 1) / steps for i in range(steps)]


def bars_from_path(path: Sequence[float], *, wick: float, digits: int) -> list[Bar]:
    """One bar per step of the path; each turning point becomes a fractal pivot."""
    return [
        Bar(
            time=EPOCH + i * H1,
            open=round(o, digits),
            high=round(max(o, c) + wick, digits),
            low=round(min(o, c) - wick, digits),
            close=round(c, digits),
        )
        for i, (o, c) in enumerate(zip(path, path[1:]))
    ]


# Two clean tests of 1.0850 and two of 1.0750, closing back at 1.0800 in the middle.
# 40 bars of H1 spanning three UTC dates, so the session window is a real subset.
EURUSD_PATH = (
    [1.0800]
    + leg(1.0800, 1.0850, 5)
    + leg(1.0850, 1.0750, 10)
    + leg(1.0750, 1.0850, 10)
    + leg(1.0850, 1.0750, 10)
    + leg(1.0750, 1.0800, 5)
)
EURUSD = bars_from_path(EURUSD_PATH, wick=WICK, digits=5)

# The same shape three thousand times bigger — XAUUSD territory. Used to prove the zone
# tolerance is ATR-relative, not a hardcoded pip count.
XAUUSD = bars_from_path([p * 3000 for p in EURUSD_PATH], wick=WICK * 3000, digits=2)
