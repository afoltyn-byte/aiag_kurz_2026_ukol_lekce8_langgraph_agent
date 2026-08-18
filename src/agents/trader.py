"""trader agent — spec §4 `trader`.

Fetches OHLC from the MT5 MCP server, derives levels with `charting.py`, renders the
annotated PNG, and has the model describe the zones. Reads `instrument` and
`timeframe`; writes `chart`, `errors`, `agent_log`.

Two things this module owns that `charting.py` deliberately does not:

* **The clock.** The timestamped filename and the retention purge live here, which is
  what lets `charting.py` stay a pure function.
* **The MT5 tool contract.** The spec does not define the server's tool names, so they
  are discovered at runtime by deterministic string matching — never by the model.

`run()` **never raises.** Every failure becomes an `errors` entry plus an `agent_log`
line and control returns to the supervisor; an escaping exception would kill the run
and lose the trace.

**A failed fetch still writes a `chart`** — a sentinel with `path: None` and an `error`
string, mirroring how `analytics` writes an empty-but-present result. The router reads a
present artifact as produced, so this is what stops a dead MT5 from being retried until
`step_limit` with nothing to show; the writer degrades to a document that says the chart
could not be generated. The one case that still writes `chart = None` is a missing
`instrument`, which is a contract violation rather than an MT5 failure — and the router
sends that state to `clarify`, never back here.

Trade-off worth knowing: a *transient* MT5 hiccup now degrades on the first failure
instead of being retried by the loop. An internal retry would be a separate change.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from src.agent.config import (
    CHART_DIR,
    CHART_RETENTION_DAYS,
    DEFAULT_TIMEFRAME,
    MT5_BAR_COUNT,
    MT5_OHLC_TOOL,
    MT5_OHLC_TOOL_CANDIDATES,
    MT5_RANGE_PADDING,
    MT5_SERVER_UTC_OFFSET_H,
    MT5_TIMEFRAME_SECONDS,
    MT5_TIMEOUT_S,
    chat_model,
    mt5_auth_headers,
    mt5_mcp_url,
)
from src.agent.state import AgentState
from src.agents.charting import Bar, Level, derive_levels, infer_digits, render_chart
from src.agents.mcp_client import call_mcp_tool, match_parameters, select_tool

AGENT = "trader"

# Parameter-name aliases for the discovered tool, most explicit first.
_SYMBOL_KEYS = ("symbol", "instrument", "pair", "ticker", "name")
_TIMEFRAME_KEYS = ("timeframe", "time_frame", "tf", "period", "interval", "resolution")
_COUNT_KEYS = ("count", "num_bars", "bars", "limit", "n", "num", "size")

# Not every MT5 MCP tool takes a bar count — plenty want an explicit window instead, and
# some declare both with the range required.
_FROM_KEYS = ("datetime_from", "date_from", "time_from", "from", "start", "start_time", "since")
_TO_KEYS = ("datetime_to", "date_to", "time_to", "to", "end", "end_time", "until")

# Where a JSON payload might keep the bar list. `history` is not speculative: it is what
# `get_chart_history` returns, alongside metadata keys like `data_available_from`.
_BARS_KEYS = ("bars", "rates", "candles", "ohlc", "history", "data", "result", "items")

# Per-field spellings inside one bar.
_BAR_FIELDS: dict[str, tuple[str, ...]] = {
    "time": ("time", "timestamp", "datetime", "date", "t"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
}


# --------------------------------------------------------------------------
# files: the clock lives here, not in charting.py
# --------------------------------------------------------------------------


def chart_path(symbol: str, timeframe: str, *, now: datetime | None = None) -> str:
    """`{CHART_DIR}/{symbol}_{timeframe}_{YYYYMMDD-HHMMSS}.png`.

    Timestamped so a retry writes a new file instead of corrupting the one a previous
    attempt half-wrote.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    Path(CHART_DIR).mkdir(parents=True, exist_ok=True)
    return str(Path(CHART_DIR) / f"{symbol}_{timeframe}_{stamp}.png")


def purge_old_charts(*, now: datetime | None = None) -> list[str]:
    """Delete PNGs older than `CHART_RETENTION_DAYS`; return what went.

    Runs on node entry (spec §4). Never raises: a chart we could not delete is not a
    reason to fail the run, so the failure is reported and the fetch continues.
    """
    directory = Path(CHART_DIR)
    if not directory.is_dir():
        return []

    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=CHART_RETENTION_DAYS)).timestamp()
    removed: list[str] = []
    for png in sorted(directory.glob("*.png")):
        try:
            if png.stat().st_mtime < cutoff:
                png.unlink()
                removed.append(str(png))
        except OSError:
            continue
    return removed


# --------------------------------------------------------------------------
# MT5 MCP: runtime discovery, no model involved
# --------------------------------------------------------------------------


def select_ohlc_tool(tool_names: Iterable[str]) -> str:
    """The OHLC tool. `MT5_OHLC_TOOL` pins an exact name; otherwise discovery."""
    return select_tool(
        tool_names,
        MT5_OHLC_TOOL_CANDIDATES,
        pinned=MT5_OHLC_TOOL,
        label="MT5 OHLC tool",
    )


def build_tool_arguments(
    input_schema: Mapping[str, Any],
    *,
    symbol: str,
    timeframe: str,
    count: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Map our values onto whatever the tool actually calls its parameters.

    Both request shapes are offered — a bar count *and* a datetime window — because MT5 MCP
    wrappers disagree about which they take, and some require the window. Whichever the
    schema declares gets filled; the rest is left alone.

    A *required* parameter we cannot match is an error rather than a guess — sending a wrong
    argument name to a trading server is worse than failing loudly.
    """
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(seconds=lookback_seconds(timeframe, count))

    return match_parameters(
        input_schema,
        [
            (_SYMBOL_KEYS, symbol),
            (_TIMEFRAME_KEYS, timeframe),
            (_COUNT_KEYS, count),
            # Callables: the encoding depends on how this server typed the property.
            (_FROM_KEYS, lambda prop: encode_time(start, prop)),
            (_TO_KEYS, lambda prop: encode_time(end, prop)),
        ],
        label="MT5 tool",
    )


def lookback_seconds(timeframe: str, count: int) -> int:
    """How far back `count` bars of `timeframe` reach, padded for closed markets.

    Wall-clock span is not bar count. 200 H1 bars cover eight-plus calendar days, two of
    which are a weekend with no bars at all; asking for exactly the nominal span would hand
    `charting` a frame too short for ATR(14), which is a ValueError rather than a smaller
    chart. `MT5_RANGE_PADDING` buys the slack. Extra bars are harmless — the level maths
    uses the whole frame and the chart simply shows more history.
    """
    per_bar = MT5_TIMEFRAME_SECONDS.get(timeframe.upper(), MT5_TIMEFRAME_SECONDS["H1"])
    return int(per_bar * count * MT5_RANGE_PADDING)


def encode_time(moment: datetime, property_schema: Mapping[str, Any]) -> Any:
    """Encode a timestamp the way this particular server declared it wants.

    The parameter is called `datetime_from` on one server and `from` on another, and it is
    a string on one and epoch seconds on the next. Both are read off the schema rather than
    guessed, for the same reason the tool name is.
    """
    declared = property_schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if "integer" in types or "number" in types:
        return int(moment.timestamp())
    # Default to ISO 8601: it is what a property named `datetime_*` implies, and what an
    # untyped schema most likely means.
    return moment.replace(microsecond=0).isoformat()


# MetaTrader renders timestamps with dots, which `fromisoformat` rejects outright. Tried
# after ISO, so a server that speaks ISO is unaffected.
_TIMESTAMP_FORMATS = (
    "%Y.%m.%d %H:%M:%S",  # MT5's own rendering
    "%Y.%m.%d %H:%M",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
)


def _parse_timestamp(text: str) -> datetime:
    cleaned = text.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass
    for pattern in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
    raise ValueError(f"unrecognised bar timestamp {text!r}")


def _to_epoch(value: Any) -> int:
    """Epoch seconds from an epoch number or a timestamp string.

    A **naive** string is broker *server* time, not UTC — MT5 has no concept of the
    difference and just prints the server clock. It is therefore stamped with
    `MT5_SERVER_UTC_OFFSET_H` rather than with UTC, which is what makes the epoch truthful.
    `derive_levels` then shifts back by the same offset to find session boundaries, so the
    two cancel and session grouping lands on server midnight either way — but leaving the
    offset unset no longer means the timestamps are quietly wrong by two or three hours.
    """
    if isinstance(value, bool):
        raise ValueError(f"bad bar timestamp: {value!r}")
    if isinstance(value, (int, float)):
        return int(value)  # already an epoch; no timezone to infer
    if isinstance(value, str):
        parsed = _parse_timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone(timedelta(hours=MT5_SERVER_UTC_OFFSET_H))
            )
        return int(parsed.timestamp())
    raise ValueError(f"bad bar timestamp: {value!r}")


def _pick(row: Mapping[str, Any], field: str) -> Any:
    lowered = {k.lower(): v for k, v in row.items()}
    for alias in _BAR_FIELDS[field]:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(f"bar is missing {field}: {sorted(row)}")


def parse_bars(payload: Any) -> list[Bar]:
    """Normalise whatever the server returned into `list[Bar]`.

    Accepts a bare list, or a dict holding the list under one of the usual keys, and
    tolerates the short key spellings MT5 wrappers like (`t/o/h/l/c`).
    """
    rows = payload
    if isinstance(rows, Mapping):
        for key in _BARS_KEYS:
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
        else:
            raise ValueError(f"no bar list in payload keys {sorted(rows)}")

    if not isinstance(rows, list) or not rows:
        raise ValueError("MT5 returned no bars")

    bars: list[Bar] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"bar is not an object: {row!r}")
        bar = Bar(
            time=_to_epoch(_pick(row, "time")),
            open=float(_pick(row, "open")),
            high=float(_pick(row, "high")),
            low=float(_pick(row, "low")),
            close=float(_pick(row, "close")),
        )
        if bar["high"] < bar["low"]:
            raise ValueError(f"bar has high < low: {row!r}")
        bars.append(bar)

    bars.sort(key=lambda b: b["time"])
    return bars


def fetch_ohlc(
    instrument: str,
    timeframe: str,
    *,
    count: int = MT5_BAR_COUNT,
    url: str | None = None,
) -> list[Bar]:
    """Fetch OHLC over MCP and normalise it. Blocking; raises on any failure."""
    endpoint = url or mt5_mcp_url()
    if not endpoint:
        raise ValueError("MT5_MCP_URL is not set in the environment")

    tool_name, payload = call_mcp_tool(
        endpoint,
        MT5_OHLC_TOOL_CANDIDATES,
        lambda schema: build_tool_arguments(
            schema, symbol=instrument, timeframe=timeframe, count=count
        ),
        timeout_s=MT5_TIMEOUT_S,
        pinned=MT5_OHLC_TOOL,
        # None when MT5_API_KEY is unset, which leaves a local unauthenticated server
        # working exactly as before.
        headers=mt5_auth_headers(),
        label="MT5 OHLC tool",
    )
    try:
        return parse_bars(payload)
    except ValueError as exc:
        # Name the tool: when discovery guesses wrong this is the one fact that
        # explains the failure.
        raise ValueError(f"{exc} (tool {tool_name!r})") from exc


# --------------------------------------------------------------------------
# the model's only job here: prose about numbers it did not compute
# --------------------------------------------------------------------------

_COMMENTARY_PROMPT = """You are a broker-side analyst describing a price chart.

Instrument: {instrument}
Timeframe: {timeframe}
Last close: {last_close}

Levels already computed from the OHLC data (do not recompute or adjust them):
{levels}

Write 2-4 sentences in English describing how price is positioned relative to these
zones and which of them matter most. Reference the numbers exactly as given. This is
analytical material for a human reader, not a recommendation: do not tell anyone to
buy, sell, or place an order, and do not add price targets of your own."""


def interpret_zones(
    instrument: str,
    timeframe: str,
    levels: Sequence[Level],
    last_close: float,
    config: Mapping[str, Any] | None = None,
) -> str:
    """A short analytical note about the zones. Model in a supportive role only.

    The levels are inputs, not outputs: the model cannot move one. See spec §4.
    """
    rendered = "\n".join(
        f"- {lv['kind']} {lv['price']} (zone {lv['low']}-{lv['high']}, "
        f"touches {lv['touches']}, strength {lv['strength']}"
        + (f", entry candidate {lv['entry']}" if lv["entry"] else "")
        + ")"
        for lv in levels
    )
    prompt = _COMMENTARY_PROMPT.format(
        instrument=instrument,
        timeframe=timeframe,
        last_close=last_close,
        levels=rendered or "- none",
    )
    response = chat_model(AGENT, config).invoke(prompt)
    return str(response.content).strip()


# --------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------


def _runtime_config() -> Mapping[str, Any] | None:
    """The invocation's `configurable`, for the per-agent model override of spec §6.

    Read through `langgraph.config.get_config()` rather than a second node parameter,
    so the entry point keeps the `(state) -> dict` shape §2 requires. Outside a graph
    run — the standalone CLI, unit tests — it raises, and `None` means "use defaults".
    """
    try:
        from langgraph.config import get_config

        return get_config()
    except Exception:
        return None


def failed_chart(symbol: str, timeframe: str, reason: str) -> dict[str, Any]:
    """The failure sentinel: same keys as a real chart, `path` empty, `error` filled.

    Present-but-empty rather than `None`, for the same reason `analytics` returns an empty
    result: the router treats a present artifact as produced, so the run moves on to a
    document that reports the gap instead of retrying until the step limit.
    """
    return {
        "path": None,
        "symbol": symbol,
        "timeframe": timeframe,
        "levels": [],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commentary": "",
        "error": reason,
    }


def run(state: AgentState) -> dict[str, Any]:
    """Fetch, derive, render, describe. Returns a partial update; never raises."""
    started = perf_counter()
    errors: list[str] = []

    # Resolving the symbol comes first and on its own: without it there is nothing to
    # write a sentinel *about*, and a missing instrument is a contract violation rather
    # than a fetch failure.
    instrument = (state.get("instrument") or "").strip().upper()
    if not instrument:
        return {
            "chart": None,
            "errors": [f"{AGENT}: no instrument resolved"],
            "agent_log": [
                {
                    "agent": AGENT,
                    "status": "error",
                    "duration_s": round(perf_counter() - started, 3),
                }
            ],
        }
    timeframe = state.get("timeframe") or DEFAULT_TIMEFRAME

    try:
        purge_old_charts()
        bars = fetch_ohlc(instrument, timeframe)

        levels = derive_levels(
            bars,
            session_offset_hours=MT5_SERVER_UTC_OFFSET_H,
            digits=infer_digits(bars),
        )

        generated_at = datetime.now(timezone.utc)
        path = chart_path(instrument, timeframe, now=generated_at)
        render_chart(bars, levels, instrument, timeframe, path)

        # Commentary is the only optional part: a chart that rendered must survive a
        # failed LLM call.
        commentary = ""
        try:
            commentary = interpret_zones(
                instrument, timeframe, levels, bars[-1]["close"], _runtime_config()
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(f"{AGENT}: commentary failed, chart kept: {exc!r}")

        update: dict[str, Any] = {
            "chart": {
                "path": path,
                "symbol": instrument,
                "timeframe": timeframe,
                "levels": levels,
                "generated_at": generated_at.isoformat(timespec="seconds"),
                "commentary": commentary,
            },
            "agent_log": [
                {
                    "agent": AGENT,
                    "status": "ok" if not errors else "partial",
                    "duration_s": round(perf_counter() - started, 3),
                }
            ],
        }
        if errors:
            update["errors"] = errors
        return update

    except Exception as exc:  # noqa: BLE001 - the node contract is "never raise"
        return {
            "chart": failed_chart(instrument, timeframe, repr(exc)),
            "errors": errors + [f"{AGENT}: {exc!r}"],
            "agent_log": [
                {
                    "agent": AGENT,
                    "status": "error",
                    "duration_s": round(perf_counter() - started, 3),
                }
            ],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the trader agent on its own.")
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--no-commentary",
        action="store_true",
        help="skip the LLM call — chart and levels only, no OPENAI_API_KEY needed",
    )
    args = parser.parse_args()

    if args.no_commentary:
        # Rebind the module attribute; run() resolves the name as a global at call
        # time. Unsetting OPENAI_API_KEY instead would make the LLM call *fail* and log
        # a spurious error, which is not what "skip" means.
        globals()["interpret_zones"] = lambda *_, **__: ""

    minimal = AgentState(instrument=args.instrument, timeframe=args.timeframe)
    print(json.dumps(run(minimal), indent=2, default=str))
