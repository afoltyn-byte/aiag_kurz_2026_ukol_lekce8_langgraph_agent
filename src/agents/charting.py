"""Level derivation and chart rendering. No LLM, and no clock.

`derive_levels` is a **pure function** of the OHLC frame: same bars in, same levels
out, on any machine. That is what makes a chart reproducible from a checkpoint and
what makes this module unit-testable against a fixed fixture. Two consequences worth
stating, because both are easy to break by accident:

* **No `datetime.now()` anywhere in this module.** The "current session" is the last
  session *present in the frame*, not the wall clock. The timestamped filename and
  the retention purge live in the `trader` node instead.
* **No plotting import at module level.** `derive_levels` must stay cheap and
  importable without matplotlib; `render_chart` imports it lazily.

Contract on bad input: this module is a helper, not an agent, so it **raises**
`ValueError`. The `trader` node catches it and records an `errors` entry. Returning
an empty level list instead would be indistinguishable from a frame that genuinely
has no levels.

See spec §4 (`trader`) for the full derivation contract.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from statistics import fmean
from typing import Literal, Sequence, TypedDict

from src.agent.config import (
    ATR_PERIOD,
    CHART_DPI,
    CHART_FIGSIZE,
    MAX_ZONES_PER_SIDE,
    PIVOT_LOOKBACK,
    ZONE_ATR_MULT,
    ZONE_STRENGTH_SATURATION,
)

LevelKind = Literal["support", "resistance", "session_high", "session_low"]
EntrySide = Literal["long", "short"]


class Bar(TypedDict):
    """One OHLC bar as the MT5 MCP server returns it. Plain dict, no pandas."""

    time: int  # epoch seconds, UTC
    open: float
    high: float
    low: float
    close: float


class Level(TypedDict):
    kind: LevelKind
    price: float  # representative price of the zone
    low: float  # zone envelope; == price for session extremes
    high: float
    touches: int
    strength: float  # 0.0-1.0, saturating at ZONE_STRENGTH_SATURATION
    entry: EntrySide | None


# --------------------------------------------------------------------------
# pure derivation
# --------------------------------------------------------------------------


def average_true_range(bars: Sequence[Bar], period: int = ATR_PERIOD) -> float:
    """Simple mean of the last `period` true ranges - not Wilder's smoothing.

    Wilder's is equally deterministic but carries the whole history in its seed, which
    makes a hand-checked fixture much harder to reason about. The value is only used
    as a "close enough" unit, so the extra fidelity buys nothing here.
    """
    if period < 1:
        raise ValueError(f"atr_period must be >= 1, got {period}")
    if len(bars) < period + 1:
        raise ValueError(f"need at least {period + 1} bars for ATR({period}), got {len(bars)}")

    ranges: list[float] = []
    for prev, cur in zip(bars[-period - 1 :], bars[-period:]):
        prev_close = prev["close"]
        ranges.append(
            max(
                cur["high"] - cur["low"],
                abs(cur["high"] - prev_close),
                abs(cur["low"] - prev_close),
            )
        )
    return fmean(ranges)


def find_pivots(bars: Sequence[Bar], lookback: int = PIVOT_LOOKBACK) -> list[float]:
    """Fractal swing highs and lows, as prices.

    `>=` to the left and `>` to the right, mirrored for lows. The asymmetry is what
    makes a flat top yield **exactly one** pivot — the rightmost bar of the plateau,
    i.e. the most recent test of that price. Strict comparisons on both sides, the
    textbook Williams fractal, would yield *zero* pivots there and quietly miss a
    double top printed at identical highs.
    """
    if lookback < 1:
        raise ValueError(f"pivot_lookback must be >= 1, got {lookback}")
    if len(bars) < 2 * lookback + 1:
        raise ValueError(
            f"need at least {2 * lookback + 1} bars for lookback {lookback}, got {len(bars)}"
        )

    pivots: list[float] = []
    for i in range(lookback, len(bars) - lookback):
        high, low = bars[i]["high"], bars[i]["low"]
        left = bars[i - lookback : i]
        right = bars[i + 1 : i + 1 + lookback]

        if all(high >= b["high"] for b in left) and all(high > b["high"] for b in right):
            pivots.append(high)
        if all(low <= b["low"] for b in left) and all(low < b["low"] for b in right):
            pivots.append(low)
    return pivots


def infer_digits(bars: Sequence[Bar], cap: int = 8) -> int:
    """Decimal precision actually present in the frame.

    Used so zone prices in the document table read `1.0852` instead of
    `1.0851999999999999`. This reports *observed* precision, not the instrument's
    declared `digits`: a frame whose prices all happen to end in zero rounds shorter,
    which is cosmetic. `cap` bounds the float-repr pathologies (`0.30000000000000004`).
    """
    digits = 0
    for bar in bars:
        for key in ("open", "high", "low", "close"):
            text = repr(float(bar[key]))  # type: ignore[literal-required]
            if "e" in text or "E" in text or "." not in text:
                continue
            digits = max(digits, len(text.split(".")[1].rstrip("0")))
    return min(digits, cap)


def _cluster(prices: Sequence[float], tolerance: float) -> list[list[float]]:
    """Greedy grouping of sorted prices while the cluster spread fits in `tolerance`.

    Highs and lows go in together: a level is a price regardless of which direction it
    was rejected from, and a price that acted as both is a stronger level, not two.
    """
    clusters: list[list[float]] = []
    for price in sorted(prices):
        if clusters and price - clusters[-1][0] <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return clusters


def _session_date(bar: Bar, shift: timedelta) -> date:
    return (datetime.fromtimestamp(bar["time"], tz=timezone.utc) + shift).date()


def _session_bars(bars: Sequence[Bar], offset_hours: float) -> list[Bar]:
    """Bars sharing the last bar's date, in broker-server time.

    An FX day does not break at midnight UTC, so the caller passes the server offset
    (MT5 is typically UTC+2/+3). The date comes from the data, never from the clock.
    """
    shift = timedelta(hours=offset_hours)
    last = _session_date(bars[-1], shift)
    return [b for b in bars if _session_date(b, shift) == last]


def _strength(touches: int) -> float:
    return round(min(1.0, touches / ZONE_STRENGTH_SATURATION), 4)


def _round(value: float, digits: int | None) -> float:
    return value if digits is None else round(value, digits)


def derive_levels(
    bars: Sequence[Bar],
    *,
    pivot_lookback: int = PIVOT_LOOKBACK,
    atr_period: int = ATR_PERIOD,
    zone_atr_mult: float = ZONE_ATR_MULT,
    max_zones_per_side: int = MAX_ZONES_PER_SIDE,
    session_offset_hours: float = 0.0,
    digits: int | None = None,
) -> list[Level]:
    """Support/resistance zones, session extremes and candidate entry levels.

    Ordered by price descending - resistances above, supports below, the way it reads
    on a chart and in the document's level table. See spec §4 for the step-by-step
    contract; the order of operations here matches it exactly.
    """
    if not bars:
        raise ValueError("bars is empty")

    atr = average_true_range(bars, atr_period)
    tolerance = zone_atr_mult * atr
    last_close = bars[-1]["close"]

    # 1-3. pivots -> zones
    zones: list[Level] = []
    for members in _cluster(find_pivots(bars, pivot_lookback), tolerance):
        price = _round(fmean(members), digits)
        kind: LevelKind = "resistance" if price > last_close else "support"
        zones.append(
            Level(
                kind=kind,
                price=price,
                low=_round(min(members), digits),
                high=_round(max(members), digits),
                touches=len(members),
                strength=_strength(len(members)),
                entry=None,
            )
        )

    # 4-5. the nearest zones on each side of the close
    supports = sorted(
        (z for z in zones if z["kind"] == "support"),
        key=lambda z: last_close - z["price"],
    )[:max_zones_per_side]
    resistances = sorted(
        (z for z in zones if z["kind"] == "resistance"),
        key=lambda z: z["price"] - last_close,
    )[:max_zones_per_side]

    # 7. entry candidates: nearest each side. Zones only - never a session extreme, so
    #    a session high coinciding with a resistance cannot be marked twice.
    if supports:
        supports[0]["entry"] = "long"
    if resistances:
        resistances[0]["entry"] = "short"

    # 6. session extremes, from the frame's own last session
    session = _session_bars(bars, session_offset_hours)
    session_high = max(b["high"] for b in session)
    session_low = min(b["low"] for b in session)
    high_touches = sum(1 for b in session if session_high - b["high"] <= tolerance)
    low_touches = sum(1 for b in session if b["low"] - session_low <= tolerance)
    extremes: list[Level] = [
        Level(
            kind="session_high",
            price=_round(session_high, digits),
            low=_round(session_high, digits),
            high=_round(session_high, digits),
            touches=high_touches,
            strength=_strength(high_touches),
            entry=None,
        ),
        Level(
            kind="session_low",
            price=_round(session_low, digits),
            low=_round(session_low, digits),
            high=_round(session_low, digits),
            touches=low_touches,
            strength=_strength(low_touches),
            entry=None,
        ),
    ]

    return sorted(supports + resistances + extremes, key=lambda z: z["price"], reverse=True)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

# The linestyle is a Literal, not a str: matplotlib's stubs accept only the known
# dash patterns, and a plain `str` is rejected.
_KIND_STYLE: dict[LevelKind, tuple[str, Literal["-", "--"]]] = {
    "resistance": ("#c0392b", "-"),
    "support": ("#1e8449", "-"),
    "session_high": ("#7f8c8d", "--"),
    "session_low": ("#7f8c8d", "--"),
}


def render_chart(
    bars: Sequence[Bar],
    levels: Sequence[Level],
    symbol: str,
    timeframe: str,
    path: str,
) -> str:
    """Write an annotated candlestick PNG to `path` and return `path`.

    The destination is an argument, not something this function invents: the caller
    owns the timestamp, so a retry writes a new file rather than corrupting one, and
    this stays clock-free.
    """
    if not bars:
        raise ValueError("bars is empty")

    # Lazy, and Agg-forced: nodes run headless, and importing pyplot against a GUI
    # backend on a server is a hang, not an error.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)

    for i, bar in enumerate(bars):
        rising = bar["close"] >= bar["open"]
        colour = "#1e8449" if rising else "#c0392b"
        ax.vlines(i, bar["low"], bar["high"], color=colour, linewidth=0.8, zorder=2)
        body_low = min(bar["open"], bar["close"])
        body_height = abs(bar["close"] - bar["open"])
        # A doji has zero body height and would render as nothing; give it a hairline.
        ax.add_patch(
            Rectangle(
                (i - 0.3, body_low),
                0.6,
                body_height or (bar["high"] - bar["low"]) * 0.02,
                facecolor=colour,
                edgecolor=colour,
                zorder=3,
            )
        )

    # Labels go in a reserved gutter to the right of the last bar. Drawn over the
    # candles they collide with price action whenever a level sits near the right
    # edge — which is exactly where the interesting levels are.
    right_edge = len(bars) - 0.5
    gutter = max(3.0, len(bars) * 0.16)

    for level in levels:
        colour, style = _KIND_STYLE[level["kind"]]
        # Bounded to the data, not axhline/axhspan across the whole axis: a full-width
        # line strikes through its own label in the gutter.
        if level["high"] > level["low"]:
            ax.fill_between(
                [-1, right_edge],
                level["low"],
                level["high"],
                color=colour,
                alpha=0.10,
                zorder=1,
            )
        ax.hlines(
            level["price"],
            -1,
            right_edge,
            colors=colour,
            linestyles=style,
            linewidth=1.0,
            alpha=0.85,
            zorder=4,
        )
        label = level["kind"].replace("_", " ")
        if level["entry"] is not None:
            label = f"{label} - entry {level['entry']}"
        ax.annotate(
            f"{label}  {level['price']:g}",
            xy=(right_edge, level["price"]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=colour,
        )

    ax.set_title(f"{symbol} {timeframe} - support / resistance zones", fontsize=11)
    ax.set_xlim(-1, right_edge + gutter)
    # Ticks stay on the data; the gutter is label space, not chart.
    ax.set_xticks(list(range(0, len(bars), max(1, len(bars) // 10))))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    ax.set_xlabel("bar index")
    ax.set_ylabel("price")
    ax.margins(y=0.05)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
