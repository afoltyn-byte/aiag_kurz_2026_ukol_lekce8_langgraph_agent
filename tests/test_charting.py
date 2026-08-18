"""`charting.py` against a fixed OHLC fixture — spec §7 ("same input, same levels").

The fixture is built from an explicit price path, so every pivot in it is there by
construction and the golden assertion below can be checked by hand. Nothing here
touches a clock, a network or a display.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from src.agents import charting
from src.agents.charting import Bar, average_true_range, derive_levels, find_pivots, render_chart
from tests.ohlc_fixture import EPOCH, EURUSD, H1, XAUUSD


# --- ATR ----------------------------------------------------------------------


def test_atr_is_the_mean_of_true_ranges() -> None:
    """Hand-computable: TR = 2.0 then 1.5, so ATR(2) = 1.75."""
    bars = [
        Bar(time=EPOCH, open=10.0, high=10.0, low=10.0, close=10.0),
        Bar(time=EPOCH + H1, open=10.0, high=11.0, low=9.0, close=10.5),
        Bar(time=EPOCH + 2 * H1, open=10.5, high=12.0, low=10.5, close=11.0),
    ]
    assert average_true_range(bars, period=2) == pytest.approx(1.75)


def test_atr_rejects_a_frame_shorter_than_its_period() -> None:
    with pytest.raises(ValueError, match="at least 15 bars"):
        average_true_range(EURUSD[:10], period=14)


def test_atr_rejects_a_non_positive_period() -> None:
    with pytest.raises(ValueError, match="atr_period must be >= 1"):
        average_true_range(EURUSD, period=0)


# --- pivots -------------------------------------------------------------------


def test_pivots_are_the_turning_points_of_the_path() -> None:
    assert find_pivots(EURUSD) == [1.0852, 1.0748, 1.0852, 1.0748]


def test_flat_top_yields_exactly_one_pivot_the_most_recent() -> None:
    """Strict-both-sides would return none here and miss the double top."""
    highs = [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]
    bars = [
        Bar(time=EPOCH + i * H1, open=h, high=h, low=h - 5.0, close=h)
        for i, h in enumerate(highs)
    ]
    assert find_pivots(bars, lookback=2).count(4.0) == 1


def test_pivots_reject_a_non_positive_lookback() -> None:
    with pytest.raises(ValueError, match="pivot_lookback must be >= 1"):
        find_pivots(EURUSD, lookback=0)


def test_pivots_reject_a_frame_narrower_than_the_window() -> None:
    with pytest.raises(ValueError, match="at least 7 bars"):
        find_pivots(EURUSD[:6], lookback=3)


# --- derive_levels: the golden ------------------------------------------------


def test_golden_levels_for_the_fixture() -> None:
    """The regression anchor. If the algorithm changes, this is what has to be
    re-read and re-approved by hand, not silently updated."""
    assert derive_levels(EURUSD, digits=5) == [
        {"kind": "resistance", "price": 1.0852, "low": 1.0852, "high": 1.0852,
         "touches": 2, "strength": 0.5, "entry": "short"},
        {"kind": "session_high", "price": 1.0802, "low": 1.0802, "high": 1.0802,
         "touches": 1, "strength": 0.25, "entry": None},
        {"kind": "session_low", "price": 1.0758, "low": 1.0758, "high": 1.0758,
         "touches": 1, "strength": 0.25, "entry": None},
        {"kind": "support", "price": 1.0748, "low": 1.0748, "high": 1.0748,
         "touches": 2, "strength": 0.5, "entry": "long"},
    ]


def test_same_input_same_levels() -> None:
    assert derive_levels(EURUSD) == derive_levels(EURUSD)


def test_repeated_calls_never_diverge() -> None:
    first = derive_levels(EURUSD, digits=5)
    assert all(derive_levels(EURUSD, digits=5) == first for _ in range(20))


def test_derivation_does_not_mutate_the_frame() -> None:
    snapshot = [dict(b) for b in EURUSD]
    derive_levels(EURUSD)
    assert [dict(b) for b in EURUSD] == snapshot


# --- derive_levels: structure -------------------------------------------------


def test_levels_are_ordered_by_price_descending() -> None:
    prices = [lv["price"] for lv in derive_levels(EURUSD)]
    assert prices == sorted(prices, reverse=True)


def test_classification_is_relative_to_the_last_close() -> None:
    last_close = EURUSD[-1]["close"]
    for level in derive_levels(EURUSD):
        if level["kind"] == "resistance":
            assert level["price"] > last_close
        elif level["kind"] == "support":
            assert level["price"] < last_close


def test_exactly_one_entry_candidate_per_side() -> None:
    levels = derive_levels(EURUSD)
    assert [lv["entry"] for lv in levels].count("long") == 1
    assert [lv["entry"] for lv in levels].count("short") == 1


def test_entry_marks_never_land_on_session_extremes() -> None:
    for level in derive_levels(EURUSD):
        if level["kind"] in ("session_high", "session_low"):
            assert level["entry"] is None


def test_zone_count_per_side_is_capped() -> None:
    levels = derive_levels(EURUSD, max_zones_per_side=1)
    assert sum(1 for lv in levels if lv["kind"] == "support") == 1
    assert sum(1 for lv in levels if lv["kind"] == "resistance") == 1


def test_zone_envelope_contains_its_representative_price() -> None:
    for level in derive_levels(EURUSD):
        assert level["low"] <= level["price"] <= level["high"]


def test_strength_saturates_within_bounds() -> None:
    for level in derive_levels(EURUSD):
        assert 0.0 < level["strength"] <= 1.0


# --- derive_levels: tolerance and session ------------------------------------


def test_tolerance_scales_with_the_instrument_not_with_pips() -> None:
    """EURUSD at 1.08 and gold at 3250 must produce the same *structure* — that is the
    whole reason the tolerance is expressed in ATRs."""
    fx = [(lv["kind"], lv["touches"]) for lv in derive_levels(EURUSD)]
    gold = [(lv["kind"], lv["touches"]) for lv in derive_levels(XAUUSD)]
    assert fx == gold


def test_a_wide_tolerance_merges_the_two_levels_into_one_zone() -> None:
    """zone_atr_mult big enough to swallow the whole 100-pip range: one cluster, and
    it now carries all four pivots."""
    levels = derive_levels(EURUSD, zone_atr_mult=100.0)
    zones = [lv for lv in levels if lv["kind"] in ("support", "resistance")]
    assert len(zones) == 1
    assert zones[0]["touches"] == 4
    assert zones[0]["low"] < zones[0]["high"]  # a real envelope, not a single price


def test_session_window_follows_the_broker_offset() -> None:
    """The last bar is 03:00Z, so a -4h server offset moves it into the previous
    session — a wider one, hence a different low."""
    utc = {lv["kind"]: lv["price"] for lv in derive_levels(EURUSD, digits=5)}
    shifted = {
        lv["kind"]: lv["price"]
        for lv in derive_levels(EURUSD, session_offset_hours=-4, digits=5)
    }
    assert shifted["session_low"] != utc["session_low"]


def test_session_extremes_bound_the_last_session() -> None:
    levels = {lv["kind"]: lv for lv in derive_levels(EURUSD)}
    assert levels["session_high"]["price"] >= levels["session_low"]["price"]


def test_digits_rounds_zone_prices() -> None:
    for level in derive_levels(EURUSD, digits=3):
        assert level["price"] == round(level["price"], 3)


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError, match="bars is empty"):
        derive_levels([])


# --- purity invariant ---------------------------------------------------------


FORBIDDEN_CALLS = frozenset(
    {"datetime.now", "datetime.utcnow", "date.today", "time.time", "time.monotonic"}
)


def _dotted_name(node: ast.AST) -> str | None:
    """"a.b.c" for an attribute chain, "a" for a bare name, None for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_the_module_contains_no_clock_and_no_randomness() -> None:
    """A wall clock in here would make the same frame yield different levels on a
    re-run, which breaks replay from a checkpoint — silently.

    Read through the AST, not the text: a grep over the source would trip over this
    module's own docstring warning about `datetime.now()`. `datetime.fromtimestamp`
    is fine — it reads a timestamp out of the data.
    """
    tree = ast.parse(Path(charting.__file__).read_text(encoding="utf-8"))

    referenced: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Name)):
            name = _dotted_name(node)
            if name:
                referenced.add(name)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert not referenced & FORBIDDEN_CALLS, f"clock leaked in: {referenced & FORBIDDEN_CALLS}"
    assert "random" not in imported


# --- rendering ----------------------------------------------------------------


def test_render_writes_a_real_png(tmp_path: Path) -> None:
    target = tmp_path / "EURUSD_H1_20260818-120000.png"
    levels = derive_levels(EURUSD, digits=5)

    returned = render_chart(EURUSD, levels, "EURUSD", "H1", str(target))

    assert returned == str(target)
    assert target.exists()
    payload = target.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(payload) > 10_000  # a blank canvas would be far smaller


def test_render_does_not_mutate_its_inputs(tmp_path: Path) -> None:
    levels: list[Any] = derive_levels(EURUSD, digits=5)
    snapshot = [dict(lv) for lv in levels]
    render_chart(EURUSD, levels, "EURUSD", "H1", str(tmp_path / "c.png"))
    assert [dict(lv) for lv in levels] == snapshot


def test_render_handles_a_doji_without_vanishing(tmp_path: Path) -> None:
    """Zero-height body: the bar still has to be drawn."""
    flat = [
        Bar(time=EPOCH + i * H1, open=1.08, high=1.0805, low=1.0795, close=1.08)
        for i in range(20)
    ]
    target = tmp_path / "doji.png"
    render_chart(flat, [], "EURUSD", "H1", str(target))
    assert target.read_bytes().startswith(b"\x89PNG")


def test_render_rejects_an_empty_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bars is empty"):
        render_chart([], [], "EURUSD", "H1", str(tmp_path / "x.png"))


# --- infer_digits -------------------------------------------------------------


def test_infer_digits_reports_observed_not_declared_precision() -> None:
    """EURUSD is a 5-digit symbol, but this fixture's prices never reach a 5th
    decimal — so 4 is the honest answer. Rounding shorter than the instrument's
    declared precision is cosmetic; claiming a precision the data lacks is not.
    """
    assert charting.infer_digits(EURUSD) == 4
    assert charting.infer_digits(XAUUSD) == 1


def test_infer_digits_on_whole_numbers_is_zero() -> None:
    bars = [Bar(time=EPOCH, open=1.0, high=2.0, low=1.0, close=2.0)]
    assert charting.infer_digits(bars) == 0


def test_infer_digits_caps_float_repr_pathologies() -> None:
    """0.1 + 0.2 reprs as 0.30000000000000004; without the cap that is 17 digits."""
    bars = [Bar(time=EPOCH, open=0.1 + 0.2, high=1.0, low=1.0, close=1.0)]
    assert charting.infer_digits(bars, cap=8) == 8


def test_inferred_digits_survive_a_round_trip_through_derive_levels() -> None:
    digits = charting.infer_digits(EURUSD)
    for level in derive_levels(EURUSD, digits=digits):
        assert level["price"] == round(level["price"], digits)
