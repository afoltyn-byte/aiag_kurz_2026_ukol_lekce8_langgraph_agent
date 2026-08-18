"""trader node — spec §4. MT5 and the LLM are stubbed; nothing here touches a network.

The interesting cases are the failure paths: the node contract is "never raises", and a
chart that rendered must survive a failed commentary.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.agents import trader
from src.agents.charting import Bar
from tests.ohlc_fixture import EURUSD

NOW = datetime(2026, 8, 18, 12, 30, 15, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def chart_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never write into the real outputs/ directory from a test."""
    target = tmp_path / "charts"
    monkeypatch.setattr(trader, "CHART_DIR", str(target))
    return target


@pytest.fixture
def stub_mt5(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trader, "fetch_ohlc", lambda *a, **k: list(EURUSD))


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trader, "interpret_zones", lambda *a, **k: "Price sits mid-range.")


# --- filenames and retention --------------------------------------------------


def test_chart_path_is_timestamped(chart_dir: Path) -> None:
    path = Path(trader.chart_path("EURUSD", "H1", now=NOW))
    assert path.name == "EURUSD_H1_20260818-123015.png"
    assert path.parent == chart_dir


def test_chart_path_creates_the_directory(chart_dir: Path) -> None:
    assert not chart_dir.exists()
    trader.chart_path("EURUSD", "H1", now=NOW)
    assert chart_dir.is_dir()


def test_a_retry_never_reuses_a_filename(chart_dir: Path) -> None:
    first = trader.chart_path("EURUSD", "H1", now=NOW)
    second = trader.chart_path("EURUSD", "H1", now=NOW + timedelta(seconds=1))
    assert first != second


def test_purge_removes_only_charts_past_retention(chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True)
    fresh = chart_dir / "EURUSD_H1_fresh.png"
    stale = chart_dir / "EURUSD_H1_stale.png"
    for png in (fresh, stale):
        png.write_bytes(b"\x89PNG")

    old = (NOW - timedelta(days=trader.CHART_RETENTION_DAYS + 1)).timestamp()
    import os

    os.utime(stale, (old, old))

    removed = trader.purge_old_charts(now=NOW)

    assert removed == [str(stale)]
    assert fresh.exists() and not stale.exists()


def test_purge_on_a_missing_directory_is_a_no_op(chart_dir: Path) -> None:
    assert trader.purge_old_charts(now=NOW) == []


# --- tool discovery -----------------------------------------------------------


def test_tool_discovery_prefers_the_earlier_candidate() -> None:
    """`copy_rates` outranks `get_bars`: candidate order, not alphabetical order."""
    assert trader.select_ohlc_tool(["get_bars", "mt5_copy_rates", "account_info"]) == (
        "mt5_copy_rates"
    )


def test_tool_discovery_is_stable_across_orderings() -> None:
    names = ["z_ohlc_fetch", "a_ohlc_fetch"]
    assert trader.select_ohlc_tool(names) == trader.select_ohlc_tool(reversed(names))


def test_tool_discovery_fails_loudly_when_nothing_matches() -> None:
    with pytest.raises(ValueError, match="no MT5 OHLC tool matching"):
        trader.select_ohlc_tool(["account_info", "order_send"])


def test_pinned_tool_name_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trader, "MT5_OHLC_TOOL", "account_history")
    assert trader.select_ohlc_tool(["copy_rates", "account_history"]) == "account_history"


def test_pinned_tool_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trader, "MT5_OHLC_TOOL", "nope")
    with pytest.raises(ValueError, match="not offered by the server"):
        trader.select_ohlc_tool(["copy_rates"])


# --- argument mapping ---------------------------------------------------------


def test_arguments_map_onto_the_servers_own_parameter_names() -> None:
    schema = {
        "properties": {"Symbol": {}, "period": {}, "num_bars": {}},
        "required": ["Symbol", "period"],
    }
    assert trader.build_tool_arguments(schema, symbol="EURUSD", timeframe="H1", count=200) == {
        "Symbol": "EURUSD",
        "period": "H1",
        "num_bars": 200,
    }


def test_unmatched_optional_parameters_are_left_alone() -> None:
    schema = {"properties": {"symbol": {}, "flags": {}}, "required": ["symbol"]}
    assert trader.build_tool_arguments(schema, symbol="XAUUSD", timeframe="H1", count=10) == {
        "symbol": "XAUUSD"
    }


def test_an_unfillable_required_parameter_is_an_error_not_a_guess() -> None:
    schema = {"properties": {"symbol": {}, "login": {}}, "required": ["symbol", "login"]}
    with pytest.raises(ValueError, match=r"required parameters \['login'\]"):
        trader.build_tool_arguments(schema, symbol="EURUSD", timeframe="H1", count=10)


# --- payload parsing ----------------------------------------------------------


def test_parse_bars_from_a_bare_list() -> None:
    payload = [{"time": 1_755_000_000, "open": 1.08, "high": 1.081, "low": 1.079, "close": 1.0805}]
    assert trader.parse_bars(payload) == [
        Bar(time=1_755_000_000, open=1.08, high=1.081, low=1.079, close=1.0805)
    ]


def test_parse_bars_from_a_nested_key() -> None:
    inner = [{"t": 1_755_000_000, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5}]
    assert trader.parse_bars({"rates": inner})[0]["high"] == 2.0


def test_parse_bars_accepts_iso_timestamps() -> None:
    payload = [{"datetime": "2025-08-12T12:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2}]
    assert trader.parse_bars(payload)[0]["time"] == 1_755_000_000


def test_parse_bars_sorts_chronologically() -> None:
    payload = [
        {"t": 200, "o": 1, "h": 2, "l": 1, "c": 2},
        {"t": 100, "o": 1, "h": 2, "l": 1, "c": 2},
    ]
    assert [b["time"] for b in trader.parse_bars(payload)] == [100, 200]


@pytest.mark.parametrize(
    "payload,match",
    [
        ([], "no bars"),
        ({"account": 1}, "no bar list"),
        ([{"o": 1, "h": 2, "l": 1, "c": 2}], "missing time"),
        ([{"t": 1, "o": 1, "h": 1, "l": 2, "c": 1}], "high < low"),
        (["not-an-object"], "not an object"),
    ],
    ids=["empty", "no-list", "missing-field", "inverted", "not-object"],
)
def test_parse_bars_rejects_junk(payload: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        trader.parse_bars(payload)


# --- the node -----------------------------------------------------------------


def test_run_produces_the_full_chart_contract(stub_mt5: None, stub_llm: None) -> None:
    update = trader.run({"instrument": "eurusd", "timeframe": "H1"})

    chart = update["chart"]
    assert set(chart) == {
        "path",
        "symbol",
        "timeframe",
        "levels",
        "generated_at",
        "commentary",
    }
    assert chart["symbol"] == "EURUSD"  # normalised to upper case
    assert chart["timeframe"] == "H1"
    assert chart["levels"]
    assert chart["commentary"] == "Price sits mid-range."
    assert Path(chart["path"]).read_bytes().startswith(b"\x89PNG")
    assert "errors" not in update


def test_run_logs_one_ok_entry(stub_mt5: None, stub_llm: None) -> None:
    entry = trader.run({"instrument": "EURUSD"})["agent_log"][0]
    assert entry["agent"] == "trader"
    assert entry["status"] == "ok"
    assert isinstance(entry["duration_s"], float)


def test_run_defaults_the_timeframe(stub_mt5: None, stub_llm: None) -> None:
    assert trader.run({"instrument": "EURUSD"})["chart"]["timeframe"] == trader.DEFAULT_TIMEFRAME


def test_run_returns_only_keys_this_agent_owns(stub_mt5: None, stub_llm: None) -> None:
    update = trader.run({"instrument": "EURUSD"})
    assert set(update) <= {"chart", "errors", "agent_log"}


def test_run_does_not_mutate_the_state(stub_mt5: None, stub_llm: None) -> None:
    state: Any = {"instrument": "EURUSD", "timeframe": "H1"}
    snapshot = dict(state)
    trader.run(state)
    assert state == snapshot


def test_a_failed_commentary_keeps_the_chart(
    stub_mt5: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the inner try/except: prose is optional, the PNG is not."""

    def boom(*_: Any, **__: Any) -> str:
        raise RuntimeError("no API key")

    monkeypatch.setattr(trader, "interpret_zones", boom)

    update = trader.run({"instrument": "EURUSD"})

    assert update["chart"]["commentary"] == ""
    assert Path(update["chart"]["path"]).exists()
    assert any("commentary failed" in e for e in update["errors"])
    assert update["agent_log"][0]["status"] == "partial"


@pytest.mark.parametrize(
    "state,fragment",
    [
        ({}, "no instrument"),
        ({"instrument": "   "}, "no instrument"),
    ],
    ids=["absent", "blank"],
)
def test_run_reports_a_missing_instrument_without_raising(
    state: Any, fragment: str, stub_llm: None
) -> None:
    update = trader.run(state)
    assert update["chart"] is None
    assert any(fragment in e for e in update["errors"])


def test_a_dead_mt5_yields_the_failure_sentinel(
    monkeypatch: pytest.MonkeyPatch, stub_llm: None
) -> None:
    """Present-but-empty, never None: the router reads a present artifact as produced, so
    the run degrades into a document reporting the gap instead of retrying to step_limit."""

    def boom(*_: Any, **__: Any) -> list[Bar]:
        raise TimeoutError("MT5 MCP did not answer")

    monkeypatch.setattr(trader, "fetch_ohlc", boom)

    update = trader.run({"instrument": "EURUSD", "timeframe": "H4"})

    chart = update["chart"]
    assert chart is not None
    assert set(chart) == {
        "path",
        "symbol",
        "timeframe",
        "levels",
        "generated_at",
        "commentary",
        "error",
    }
    assert chart["path"] is None
    assert chart["levels"] == []
    assert chart["symbol"] == "EURUSD"
    assert chart["timeframe"] == "H4"  # kept, so the report can still name what failed
    assert "MT5 MCP did not answer" in chart["error"]
    assert any("MT5 MCP did not answer" in e for e in update["errors"])
    assert update["agent_log"][0]["status"] == "error"


def test_unparseable_mt5_output_also_yields_the_sentinel(
    monkeypatch: pytest.MonkeyPatch, stub_llm: None
) -> None:
    monkeypatch.setattr(trader, "fetch_ohlc", lambda *a, **k: trader.parse_bars([]))
    update = trader.run({"instrument": "EURUSD"})
    assert update["chart"]["error"]
    assert update["errors"]


def test_fetch_without_a_configured_url_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MT5_MCP_URL", raising=False)
    with pytest.raises(ValueError, match="MT5_MCP_URL is not set"):
        trader.fetch_ohlc("EURUSD", "H1")


def test_run_survives_a_frame_too_short_to_derive_levels(
    monkeypatch: pytest.MonkeyPatch, stub_llm: None
) -> None:
    """charting raises on a 3-bar frame; the node absorbs it into the sentinel."""
    monkeypatch.setattr(trader, "fetch_ohlc", lambda *a, **k: list(EURUSD[:3]))
    update = trader.run({"instrument": "EURUSD"})
    assert update["chart"]["path"] is None
    assert any("ATR" in e or "bars" in e for e in update["errors"])


def test_a_missing_instrument_still_writes_chart_none(stub_llm: None) -> None:
    """Not a fetch failure but a contract violation, and there is nothing to write a
    sentinel *about*. The router sends this state to clarify, never back here."""
    update = trader.run({})
    assert update["chart"] is None
    assert any("no instrument" in e for e in update["errors"])


def test_the_sentinel_is_a_fresh_object_each_time() -> None:
    """A shared mutable default in state would alias across runs."""
    first = trader.failed_chart("EURUSD", "H1", "boom")
    second = trader.failed_chart("EURUSD", "H1", "boom")
    assert first is not second


# --- commentary prompt --------------------------------------------------------


def test_commentary_prompt_carries_the_levels_and_forbids_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model must see the computed numbers and be told this is not a recommendation
    (spec §1 non-goals)."""
    captured: dict[str, str] = {}

    class FakeModel:
        def invoke(self, prompt: str) -> Any:
            captured["prompt"] = prompt
            return type("R", (), {"content": "  note  "})()

    monkeypatch.setattr(trader, "chat_model", lambda *a, **k: FakeModel())

    from src.agents.charting import derive_levels

    levels = derive_levels(EURUSD, digits=5)
    result = trader.interpret_zones("EURUSD", "H1", levels, EURUSD[-1]["close"])

    assert result == "note"  # stripped
    assert "1.0852" in captured["prompt"]
    assert "entry candidate short" in captured["prompt"]
    assert "not a recommendation" in captured["prompt"]


def test_cli_json_dump_is_serialisable(stub_mt5: None, stub_llm: None) -> None:
    """The __main__ block prints the update as JSON; make sure it can."""
    json.dumps(trader.run({"instrument": "EURUSD"}), default=str)


# --- MT5 auth -----------------------------------------------------------------


def _capture_call(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Intercept the MCP call and keep the kwargs it was given."""
    captured: dict[str, Any] = {}

    def fake(url: Any, candidates: Any, build: Any, **kwargs: Any) -> Any:
        captured["url"] = url
        captured.update(kwargs)
        return "copy_rates", [
            {"time": 1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
        ]

    monkeypatch.setattr(trader, "call_mcp_tool", fake)
    monkeypatch.setenv("MT5_MCP_URL", "http://127.0.0.1:8000/mcp")
    return captured


def test_no_auth_header_is_sent_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local unauthenticated server must behave exactly as it did before the key
    existed \u2014 `None`, not an empty dict, so no HTTP client is built for it."""
    captured = _capture_call(monkeypatch)
    monkeypatch.delenv("MT5_API_KEY", raising=False)

    trader.fetch_ohlc("EURUSD", "H1")

    assert captured["headers"] is None


def test_the_key_travels_as_a_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_call(monkeypatch)
    monkeypatch.setenv("MT5_API_KEY", "s3cret")

    trader.fetch_ohlc("EURUSD", "H1")

    assert captured["headers"] == {"Authorization": "Bearer s3cret"}


def test_a_blank_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_call(monkeypatch)
    monkeypatch.setenv("MT5_API_KEY", "")

    trader.fetch_ohlc("EURUSD", "H1")

    assert captured["headers"] is None


def test_the_header_shape_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCP has no convention for this; X-API-Key with a raw value is a real variant."""
    from src.agent import config

    monkeypatch.setattr(config, "MT5_AUTH_HEADER", "X-API-Key")
    monkeypatch.setattr(config, "MT5_AUTH_SCHEME", "")
    monkeypatch.setenv("MT5_API_KEY", "s3cret")

    assert config.mt5_auth_headers() == {"X-API-Key": "s3cret"}


def test_the_key_is_never_put_in_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in a URL ends up in logs and proxy history."""
    captured = _capture_call(monkeypatch)
    monkeypatch.setenv("MT5_API_KEY", "s3cret")

    trader.fetch_ohlc("EURUSD", "H1")

    assert "s3cret" not in captured["url"]


# --- the MT5 datetime range ---------------------------------------------------


RANGE_NOW = datetime(2026, 8, 18, 14, 30, tzinfo=timezone.utc)


def test_the_shape_that_failed_against_the_real_server_now_works() -> None:
    """First real run: `cannot fill required parameters ['datetime_from', 'datetime_to']`.
    That server takes a window, not a bar count."""
    schema = {
        "properties": {
            "symbol": {},
            "timeframe": {},
            "datetime_from": {"type": "string"},
            "datetime_to": {"type": "string"},
        },
        "required": ["symbol", "timeframe", "datetime_from", "datetime_to"],
    }

    arguments = trader.build_tool_arguments(
        schema, symbol="EURUSD", timeframe="H1", count=200, now=RANGE_NOW
    )

    assert arguments["symbol"] == "EURUSD"
    assert arguments["datetime_to"] == "2026-08-18T14:30:00+00:00"
    assert arguments["datetime_from"] < arguments["datetime_to"]


def test_a_timestamp_typed_integer_is_sent_as_epoch_seconds() -> None:
    """Read off the schema rather than guessed, for the same reason the tool name is."""
    schema = {"properties": {"symbol": {}, "from": {"type": "integer"}, "to": {"type": "integer"}}}

    arguments = trader.build_tool_arguments(
        schema, symbol="EURUSD", timeframe="H1", count=200, now=RANGE_NOW
    )

    assert arguments["to"] == int(RANGE_NOW.timestamp())
    assert isinstance(arguments["from"], int)


def test_a_tool_taking_a_count_is_unaffected() -> None:
    schema = {"properties": {"symbol": {}, "timeframe": {}, "count": {}}, "required": ["symbol"]}
    arguments = trader.build_tool_arguments(
        schema, symbol="EURUSD", timeframe="H1", count=200, now=RANGE_NOW
    )
    assert arguments == {"symbol": "EURUSD", "timeframe": "H1", "count": 200}


def test_a_tool_declaring_both_shapes_gets_both() -> None:
    """Some wrappers want a window *and* a cap; filling both is harmless."""
    schema = {
        "properties": {"symbol": {}, "count": {}, "datetime_from": {}, "datetime_to": {}},
        "required": ["symbol"],
    }
    arguments = trader.build_tool_arguments(
        schema, symbol="EURUSD", timeframe="H1", count=200, now=RANGE_NOW
    )
    assert set(arguments) == {"symbol", "count", "datetime_from", "datetime_to"}


# --- the lookback window ------------------------------------------------------


def test_the_window_is_padded_beyond_the_nominal_span() -> None:
    """Markets close. 200 H1 bars span eight-plus calendar days, two of which are a weekend
    with no bars, so the nominal span would hand charting a frame too short for ATR(14)."""
    nominal = 200 * 3600
    assert trader.lookback_seconds("H1", 200) > nominal


def test_the_window_scales_with_the_timeframe() -> None:
    assert trader.lookback_seconds("H4", 200) == 4 * trader.lookback_seconds("H1", 200)


def test_an_unknown_timeframe_falls_back_to_an_hour_per_bar() -> None:
    assert trader.lookback_seconds("H3", 200) == trader.lookback_seconds("H1", 200)


def test_the_timeframe_is_matched_case_insensitively() -> None:
    assert trader.lookback_seconds("h4", 10) == trader.lookback_seconds("H4", 10)


def test_encode_time_defaults_to_iso_for_an_untyped_property() -> None:
    """A property named `datetime_from` with no declared type most likely means a string."""
    assert trader.encode_time(RANGE_NOW, {}) == "2026-08-18T14:30:00+00:00"


def test_encode_time_drops_microseconds() -> None:
    moment = RANGE_NOW.replace(microsecond=123456)
    assert trader.encode_time(moment, {"type": "string"}).endswith("14:30:00+00:00")


def test_bars_nested_under_history_are_found() -> None:
    """`get_chart_history` returns them under `history`, next to metadata keys like
    `data_available_from` — found on a live run against a real MT5 MCP server."""
    payload = {
        "symbol": "EURUSD",
        "period": "H1",
        "data_available_from": "2020-01-01",
        "history": [
            {"time": 1_755_000_000, "open": 1.08, "high": 1.0812, "low": 1.0798, "close": 1.081}
        ],
    }
    assert trader.parse_bars(payload)[0]["close"] == 1.081


def test_metadata_keys_are_not_mistaken_for_the_bar_list() -> None:
    """`data_available_from` starts with `data`, which must not match the `data` key."""
    with pytest.raises(ValueError, match="no bar list"):
        trader.parse_bars({"symbol": "EURUSD", "data_available_from": "2020-01-01"})


# --- timestamp formats --------------------------------------------------------


def server_epoch(year: int, month: int, day: int, hour: int = 0) -> int:
    """A wall-clock moment in *server* time as a true epoch.

    Computed from the configured offset rather than hardcoded: a naive MT5 timestamp is
    server time, so the epoch it maps to legitimately moves when a deployment sets
    `MT5_SERVER_UTC_OFFSET_H`. A literal here would fail the moment someone configures their
    own broker, which is not a defect worth reporting.
    """
    zone = timezone(timedelta(hours=trader.MT5_SERVER_UTC_OFFSET_H))
    return int(datetime(year, month, day, hour, tzinfo=zone).timestamp())


def test_mt5_dotted_timestamps_parse_like_the_dashed_ones() -> None:
    """MetaTrader prints `2026.08.06 06:00:00`, which `fromisoformat` rejects outright —
    the exact failure from a live run against a real MT5 MCP server."""
    assert trader._to_epoch("2026.08.06 06:00:00") == trader._to_epoch("2026-08-06 06:00:00")


@pytest.mark.parametrize(
    "text",
    [
        "2026.08.06 06:00:00",
        "2026.08.06 06:00",
        "2026-08-06 06:00:00",
        "2026-08-06T06:00:00",
        "2026/08/06 06:00:00",
        "06.08.2026 06:00:00",
        "  2026.08.06 06:00:00  ",
    ],
    ids=["mt5", "mt5-no-secs", "dashed", "iso", "slashed", "day-first", "padded"],
)
def test_every_naive_spelling_lands_on_the_same_instant(text: str) -> None:
    """All of these are server time; the aware form is covered separately below."""
    assert trader._to_epoch(text) == server_epoch(2026, 8, 6, 6)


def test_a_date_without_a_time_is_midnight() -> None:
    """D1 bars sometimes arrive as a bare date."""
    assert trader._to_epoch("2026.08.06") == trader._to_epoch("2026.08.06 00:00:00")


def test_an_epoch_number_passes_through_untouched() -> None:
    """Already absolute: there is no timezone to infer, so nothing is applied."""
    assert trader._to_epoch(1_755_000_000) == 1_755_000_000


def test_an_unrecognised_timestamp_says_so_with_the_value() -> None:
    with pytest.raises(ValueError, match="unrecognised bar timestamp 'yesterday'"):
        trader._to_epoch("yesterday")


def test_a_naive_timestamp_is_read_as_broker_server_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MT5 prints the server clock with no timezone. Treating that as UTC leaves every bar
    two or three hours off; the offset is what makes the epoch truthful. 06:00 on a UTC+3
    server is 03:00 UTC."""
    monkeypatch.setattr(trader, "MT5_SERVER_UTC_OFFSET_H", 3.0)

    assert trader._to_epoch("2026.08.06 06:00:00") == 1_785_996_000 - 3 * 3600


def test_the_same_wall_clock_moves_with_the_configured_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two servers printing the same string mean two different instants, and that is the
    whole point of the setting."""
    monkeypatch.setattr(trader, "MT5_SERVER_UTC_OFFSET_H", 0.0)
    at_utc = trader._to_epoch("2026.08.06 06:00:00")
    monkeypatch.setattr(trader, "MT5_SERVER_UTC_OFFSET_H", 3.0)
    at_eest = trader._to_epoch("2026.08.06 06:00:00")

    assert at_utc - at_eest == 3 * 3600


def test_an_explicit_zone_overrides_the_server_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that bothers to send an offset is telling the truth; do not second-guess it."""
    monkeypatch.setattr(trader, "MT5_SERVER_UTC_OFFSET_H", 3.0)
    assert trader._to_epoch("2026-08-06T06:00:00+00:00") == 1_785_996_000


def test_session_grouping_is_unaffected_by_the_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two shifts cancel: stamping the bar and then shifting back for the session date.
    Sessions land on server midnight whatever the offset is set to."""
    from src.agents.charting import _session_bars

    rows = [
        {"time": f"2026.08.06 {hour:02d}:00:00", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
        for hour in range(20, 24)
    ] + [
        {"time": f"2026.08.07 {hour:02d}:00:00", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
        for hour in range(0, 3)
    ]

    def session_size(offset: float) -> int:
        monkeypatch.setattr(trader, "MT5_SERVER_UTC_OFFSET_H", offset)
        bars = trader.parse_bars(list(rows))
        return len(_session_bars(bars, offset))

    assert session_size(0.0) == session_size(3.0) == 3  # the three bars dated 08-07
