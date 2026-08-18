"""Router branch coverage — spec §5, and spec §7 ("12 cases minimum").

Every case is a hand-built state dict in, one label out. The router is pure, so
these need no graph, no checkpointer, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agent.config import MAX_CLARIFICATIONS, MAX_STEPS
from src.agent.graph import ROUTE_MAP
from src.agent.supervisor import route_from_supervisor

CHART: dict[str, Any] = {"path": "outputs/charts/EURUSD_H1_20260818-120000.png"}
NEWS: dict[str, Any] = {"summary": "...", "items": [{"title": "t", "url": "u"}]}
DOC: dict[str, Any] = {"path": "outputs/reports/EURUSD_20260818-120000.docx"}

# analytics signals failure with an empty-but-present result (spec §4), so the
# artifact still counts as produced and the run does not loop on a dead Tavily.
NEWS_FAILED: dict[str, Any] = {"summary": "", "items": []}


def state(**kwargs: Any) -> dict[str, Any]:
    """A resolved, mid-run state. Individual cases override what they exercise."""
    base: dict[str, Any] = {
        "request": "co je nového na eurodolaru",
        "instrument": "EURUSD",
        "timeframe": "H1",
        "language": "cs",
        "scope": "news",
        "next_agent": None,
        "step_count": 1,
        "clarify_count": 0,
        "analytics_result": None,
        "chart": None,
        "document": None,
    }
    base.update(kwargs)
    return base


# --- rule 1: step limit -------------------------------------------------------


def test_step_limit_at_threshold() -> None:
    assert route_from_supervisor(state(step_count=MAX_STEPS)) == "step_limit"


def test_step_limit_beats_a_finished_document() -> None:
    """Rule order: the step check runs before the document check."""
    assert route_from_supervisor(state(step_count=MAX_STEPS + 3, document=DOC)) == "step_limit"


def test_one_step_below_the_limit_still_routes_normally() -> None:
    assert route_from_supervisor(state(step_count=MAX_STEPS - 1)) == "analytics"


# --- rule 2: done -------------------------------------------------------------


def test_document_present_is_done() -> None:
    assert route_from_supervisor(state(document=DOC)) == "done"


# --- rule 3: clarify ----------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"instrument": None},
        {"instrument": ""},
        {"scope": None},
        {"scope": "everything"},  # an LLM can return this despite the annotation
    ],
    ids=["instrument-none", "instrument-empty", "scope-none", "scope-invalid"],
)
def test_unresolved_request_goes_to_clarify(override: dict[str, Any]) -> None:
    assert route_from_supervisor(state(**override)) == "clarify"


def test_clarify_cap_ends_the_run_instead_of_asking_again() -> None:
    unresolved = state(instrument=None, clarify_count=MAX_CLARIFICATIONS)
    assert route_from_supervisor(unresolved) == "done"


def test_clarify_cap_applies_to_invalid_scope_too() -> None:
    unresolved = state(scope=None, clarify_count=MAX_CLARIFICATIONS + 1)
    assert route_from_supervisor(unresolved) == "done"


def test_last_clarification_is_still_allowed() -> None:
    unresolved = state(instrument=None, clarify_count=MAX_CLARIFICATIONS - 1)
    assert route_from_supervisor(unresolved) == "clarify"


# --- rule 4: missing artifacts ------------------------------------------------


@pytest.mark.parametrize(
    "scope,expected",
    [("news", "analytics"), ("chart", "trader"), ("both", "analytics")],
)
def test_first_missing_artifact_for_each_scope(scope: str, expected: str) -> None:
    assert route_from_supervisor(state(scope=scope)) == expected


def test_both_takes_trader_once_news_is_in() -> None:
    assert route_from_supervisor(state(scope="both", analytics_result=NEWS)) == "trader"


def test_both_takes_analytics_once_the_chart_is_in() -> None:
    assert route_from_supervisor(state(scope="both", chart=CHART)) == "analytics"


def test_llm_keeps_the_ordering_choice_within_scope() -> None:
    """`both` needs each of them; which comes first is the supervisor's judgement."""
    assert route_from_supervisor(state(scope="both", next_agent="trader")) == "trader"


@pytest.mark.parametrize(
    "scope,proposal,expected",
    [
        ("news", "trader", "analytics"),  # out of scope
        ("chart", "analytics", "trader"),  # out of scope
        ("both", "writer", "analytics"),  # premature
        ("news", "done", "analytics"),  # cannot finish without a document
        ("news", "clarify", "analytics"),  # already resolved, nothing to ask
    ],
    ids=["news-proposes-trader", "chart-proposes-analytics", "both-proposes-writer",
         "news-proposes-done", "news-proposes-clarify"],
)
def test_out_of_scope_proposal_is_overridden(scope: str, proposal: str, expected: str) -> None:
    assert route_from_supervisor(state(scope=scope, next_agent=proposal)) == expected


def test_failed_analytics_still_counts_as_produced() -> None:
    """Empty result, not None — so a dead Tavily yields a document noting the gap
    rather than an endless retry loop."""
    assert route_from_supervisor(state(scope="news", analytics_result=NEWS_FAILED)) == "writer"


def test_an_absent_chart_still_routes_to_the_trader() -> None:
    """`None` means the trader has not run yet."""
    assert route_from_supervisor(state(scope="chart", chart=None)) == "trader"


def test_the_trader_failure_sentinel_counts_as_produced() -> None:
    """The whole reason the sentinel exists: present-but-empty moves the run on to the
    writer instead of retrying a dead MT5 until the step limit."""
    sentinel = {"path": None, "symbol": "EURUSD", "levels": [], "error": "boom"}
    assert route_from_supervisor(state(scope="chart", chart=sentinel)) == "writer"


def test_both_scope_completes_on_two_sentinels() -> None:
    """Both dependencies down: the run still reaches the writer and produces a report that
    reports the gaps, rather than ending at step_limit with nothing."""
    degraded = state(
        scope="both",
        analytics_result=NEWS_FAILED,
        chart={"path": None, "levels": [], "error": "boom"},
    )
    assert route_from_supervisor(degraded) == "writer"


# --- rule 5: writer -----------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"scope": "news", "analytics_result": NEWS},
        {"scope": "chart", "chart": CHART},
        {"scope": "both", "analytics_result": NEWS, "chart": CHART},
    ],
    ids=["news", "chart", "both"],
)
def test_complete_inputs_route_to_writer(override: dict[str, Any]) -> None:
    assert route_from_supervisor(state(**override)) == "writer"


def test_writer_is_not_skipped_even_if_the_llm_says_done() -> None:
    ready = state(scope="both", analytics_result=NEWS, chart=CHART, next_agent="done")
    assert route_from_supervisor(ready) == "writer"


# --- invariants ---------------------------------------------------------------


def test_missing_keys_default_rather_than_raise() -> None:
    """A first visit has no step_count/clarify_count yet."""
    assert route_from_supervisor({"request": "x"}) == "clarify"


def test_router_does_not_mutate_its_input() -> None:
    before = state(scope="both")
    snapshot = dict(before)
    route_from_supervisor(before)
    assert before == snapshot


def test_router_is_deterministic() -> None:
    fixed = state(scope="both", next_agent="trader")
    assert len({route_from_supervisor(fixed) for _ in range(25)}) == 1


@pytest.mark.parametrize(
    "case",
    [
        state(step_count=MAX_STEPS),
        state(document=DOC),
        state(instrument=None),
        state(instrument=None, clarify_count=MAX_CLARIFICATIONS),
        state(scope="chart"),
        state(scope="both", analytics_result=NEWS),
        state(scope="news", analytics_result=NEWS),
    ],
)
def test_every_returned_label_has_an_edge(case: dict[str, Any]) -> None:
    """A label with no entry in ROUTE_MAP is a branch that silently never fires."""
    assert route_from_supervisor(case) in ROUTE_MAP
