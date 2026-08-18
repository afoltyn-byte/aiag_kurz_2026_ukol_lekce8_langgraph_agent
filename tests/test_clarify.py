"""clarify node — spec §4. `interrupt()` is stubbed to stand in for a resumed run.

The test that matters most is the last one: `interrupt()` suspends by raising, so a
blanket `except` in this node would turn human-in-the-loop into a silent no-op that
answers a question nobody was ever asked.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp

from src.agent import clarify


def stub_interrupt(monkeypatch: pytest.MonkeyPatch, resume: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake(payload: Any) -> Any:
        captured["payload"] = payload
        return resume

    monkeypatch.setattr(clarify, "interrupt", fake)
    return captured


def base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "request": "co je nového na eurodolaru",
        "instrument": None,
        "scope": None,
        "language": "cs",
    }
    state.update(overrides)
    return state


# --- what is missing ----------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"instrument": None, "scope": None}, ["instrument", "scope"]),
        ({"instrument": "EURUSD", "scope": None}, ["scope"]),
        ({"instrument": None, "scope": "news"}, ["instrument"]),
        ({"instrument": "EURUSD", "scope": "news"}, []),
        ({"instrument": "", "scope": "everything"}, ["instrument", "scope"]),
    ],
    ids=["both", "scope-only", "instrument-only", "nothing", "blank-and-invalid"],
)
def test_missing_fields_matches_the_routers_predicate(
    state: dict[str, Any], expected: list[str]
) -> None:
    """If these two disagreed, the router could send us here with nothing to ask."""
    assert clarify.missing_fields(state) == expected


# --- exactly one question, in the right language ------------------------------


def test_two_missing_fields_still_yield_a_single_question() -> None:
    question = clarify.question_for(["instrument", "scope"], "cs")
    assert question.count("?") == 1


@pytest.mark.parametrize(
    "language,fragment",
    [("cs", "Který instrument"), ("en", "Which instrument")],
)
def test_the_question_is_asked_in_the_detected_language(
    language: str, fragment: str
) -> None:
    assert fragment in clarify.question_for(["instrument"], language)


def test_an_unsupported_language_falls_back_to_english() -> None:
    """The template table is Czech + English; anything else gets English rather than a
    KeyError or an empty prompt."""
    assert clarify.question_for(["scope"], "ja") == (
        clarify.QUESTION_TEMPLATES["en"]["scope"]
    )


def test_the_scope_question_offers_all_three_options() -> None:
    for language in ("cs", "en"):
        question = clarify.question_for(["scope"], language).lower()
        assert "both" in question or "obojí" in question


def test_nothing_missing_still_produces_a_question() -> None:
    """Defensive: interrupting with an empty prompt would be worse than a broad ask."""
    assert clarify.question_for([], "en")


# --- the payload --------------------------------------------------------------


def test_the_payload_carries_the_question_and_what_was_understood() -> None:
    payload = clarify.build_payload(base_state(instrument="EURUSD", timeframe="H4"))

    assert payload["missing"] == ["scope"]
    assert payload["question"] == clarify.QUESTION_TEMPLATES["cs"]["scope"]
    assert payload["understood"] == {
        "instrument": "EURUSD",
        "timeframe": "H4",
        "scope": None,
    }
    assert payload["request"] == "co je nového na eurodolaru"
    assert payload["language"] == "cs"


def test_the_attempt_number_counts_from_one() -> None:
    assert clarify.build_payload(base_state())["attempt"] == 1
    assert clarify.build_payload(base_state(clarify_count=1))["attempt"] == 2


def test_building_the_payload_is_pure_and_repeatable() -> None:
    """The node re-executes on resume, so this runs twice per clarification."""
    state = base_state()
    first = clarify.build_payload(state)
    assert clarify.build_payload(state) == first


def test_payload_language_falls_back_when_nothing_hints_otherwise() -> None:
    """With no detected language *and* nothing Czech-looking in the request, English."""
    payload = clarify.build_payload(base_state(language=None, request="anything at all"))
    assert payload["language"] == clarify.FALLBACK_LANGUAGE


def test_payload_language_prefers_the_request_over_the_fallback() -> None:
    """The improvement: a Czech request no longer gets an English question just because the
    supervisor's own model call failed before it could write `language`."""
    payload = clarify.build_payload(base_state(language=None))
    assert payload["language"] == "cs"


# --- the answer ---------------------------------------------------------------


def test_the_documented_resume_shape_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_interrupt(monkeypatch, {"answer": "EURUSD, jen zprávy"})

    update = clarify.run(base_state())

    assert isinstance(update["messages"][0], AIMessage)
    assert isinstance(update["messages"][1], HumanMessage)
    assert update["messages"][1].content == "EURUSD, jen zprávy"


def test_a_bare_string_resume_is_accepted_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Command(resume="EURUSD")` instead of `resume={"answer": ...}` is the obvious
    mistake to make, and rejecting it would waste a clarification."""
    stub_interrupt(monkeypatch, "EURUSD")
    assert clarify.run(base_state())["messages"][1].content == "EURUSD"


def test_the_answer_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_interrupt(monkeypatch, {"answer": "  EURUSD  "})
    assert clarify.run(base_state())["messages"][1].content == "EURUSD"


@pytest.mark.parametrize("resume", [None, {"answer": None}, {"answer": "   "}, {}])
def test_an_empty_answer_records_only_the_question(
    resume: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It still burns one clarification, so rule 3 ends the run rather than looping."""
    stub_interrupt(monkeypatch, resume)

    update = clarify.run(base_state())

    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], AIMessage)
    assert update["clarify_count"] == 1


# --- the node contract --------------------------------------------------------


def test_the_counter_advances_from_state_not_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_interrupt(monkeypatch, {"answer": "EURUSD"})
    assert clarify.run(base_state(clarify_count=1))["clarify_count"] == 2


def test_run_returns_only_keys_this_node_owns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §4: `clarify_count` and `messages`. It decides nothing, so it writes no
    `instrument`, no `scope`, no `next_agent`."""
    stub_interrupt(monkeypatch, {"answer": "EURUSD"})
    assert set(clarify.run(base_state())) == {"clarify_count", "messages"}


def test_run_does_not_mutate_the_state(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_interrupt(monkeypatch, {"answer": "EURUSD"})
    state = base_state(clarify_count=1)
    snapshot = dict(state)
    clarify.run(state)
    assert state == snapshot


def test_the_question_the_caller_saw_is_the_one_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = stub_interrupt(monkeypatch, {"answer": "EURUSD"})
    update = clarify.run(base_state())
    assert update["messages"][0].content == captured["payload"]["question"]


# --- the mechanism ------------------------------------------------------------


def test_the_suspend_propagates_instead_of_being_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`interrupt()` suspends the run by raising a GraphBubbleUp. If this node ever grows
    a blanket `except Exception`, human-in-the-loop becomes a silent no-op: the graph would
    carry on with a clarification nobody was ever asked."""

    def suspend(_: Any) -> Any:
        raise GraphBubbleUp("suspend")

    monkeypatch.setattr(clarify, "interrupt", suspend)

    with pytest.raises(GraphBubbleUp):
        clarify.run(base_state())


# --- the language of last resort ----------------------------------------------


def test_a_detected_language_always_wins() -> None:
    """The supervisor owns this; the probe below must never override it."""
    state = base_state(language="en", request="co je nového na eurodolaru")
    assert clarify.language_for(state) == "en"


def test_a_czech_request_is_recognised_when_the_supervisor_detected_nothing() -> None:
    """A live run asked a Czech user in English: the supervisor's model call failed on the
    first visit, so it wrote no `language`, and this node has no model of its own."""
    state = {"request": "co je nového na eurodolaru a jak vypadá graf"}
    assert clarify.language_for(state) == "cs"
    assert "Který instrument" in clarify.build_payload(state)["question"]


@pytest.mark.parametrize(
    "request_text",
    ["what is new on eurusd", "co je noveho na eurodolaru", "", "EURUSD"],
    ids=["english", "czech-without-diacritics", "empty", "symbol-only"],
)
def test_anything_else_falls_back_to_english(request_text: str) -> None:
    """Czech typed without diacritics gets English — the honest limit of a two-language
    table, and better than guessing from word shapes."""
    assert clarify.language_for({"request": request_text}) == clarify.FALLBACK_LANGUAGE


def test_the_probe_is_not_a_language_detector() -> None:
    """It recognises Czech and nothing else; a French request is not claimed as Czech."""
    assert clarify.language_for({"request": "quoi de neuf"}) == clarify.FALLBACK_LANGUAGE
