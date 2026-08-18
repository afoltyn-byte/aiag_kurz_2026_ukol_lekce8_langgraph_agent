"""supervisor node — spec §4. The model is stubbed.

Router coverage lives in `test_router.py`; this file is about the node: what it writes,
what it refuses to trust from the model, and the two invariants that keep the graph
terminating.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent import supervisor
from src.agent.config import DEFAULT_TIMEFRAME, MAX_CLARIFICATIONS, MAX_STEPS

DECISION: dict[str, Any] = {
    "instrument": "EURUSD",
    "timeframe": "H1",
    "language": "cs",
    "scope": "both",
    "next_agent": "analytics",
    "reason": "news and chart both requested",
}


def stub_decision(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> dict[str, Any]:
    decision = {**DECISION, **overrides}
    monkeypatch.setattr(supervisor, "decide", lambda *a, **k: dict(decision))
    return decision


def base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"request": "co je nového na eurodolaru a jak vypadá graf"}
    state.update(overrides)
    return state


# --- the prompt and the schema ------------------------------------------------


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class FakeStructured:
        def invoke(self, prompt: str) -> dict[str, Any]:
            captured["prompt"] = prompt
            return dict(DECISION)

    class FakeModel:
        def with_structured_output(self, schema: Any) -> FakeStructured:
            captured["schema"] = schema
            return FakeStructured()

    monkeypatch.setattr(supervisor, "chat_model", lambda *a, **k: FakeModel())
    return captured


def test_the_prompt_carries_the_request_state_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture(monkeypatch)

    supervisor.decide(
        base_state(
            instrument="EURUSD",
            scope="both",
            analytics_result={"summary": "s", "items": []},
            chart=None,
        )
    )

    prompt = captured["prompt"]
    assert "co je nového na eurodolaru" in prompt
    assert "news analysis: yes" in prompt
    assert "chart:         no" in prompt
    assert "Guessing is worse than asking" in prompt


def test_the_conversation_is_replayed_so_a_clarification_is_picked_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4: on later visits the supervisor re-reads `messages`."""
    from langchain_core.messages import AIMessage, HumanMessage

    captured = _capture(monkeypatch)

    supervisor.decide(
        base_state(
            messages=[
                AIMessage(content="Který instrument?"),
                HumanMessage(content="EURUSD, jen zprávy"),
            ]
        )
    )

    assert "EURUSD, jen zprávy" in captured["prompt"]


def test_an_empty_conversation_is_labelled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    supervisor.decide(base_state())
    assert "nothing yet" in captured["prompt"]


def test_scope_is_offered_to_the_model_as_a_nullable_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`null` has to be reachable, or the model cannot say "I do not know"."""
    captured = _capture(monkeypatch)
    supervisor.decide(base_state())

    scope = captured["schema"]["properties"]["scope"]
    assert None in scope["enum"]
    assert "null" in scope["type"]


def test_a_non_dict_decision_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStructured:
        def invoke(self, prompt: str) -> Any:
            return ["not", "a", "dict"]

    class FakeModel:
        def with_structured_output(self, schema: Any) -> FakeStructured:
            return FakeStructured()

    monkeypatch.setattr(supervisor, "chat_model", lambda *a, **k: FakeModel())

    with pytest.raises(ValueError, match="expected a decision dict"):
        supervisor.decide(base_state())


# --- what the node writes -----------------------------------------------------


def test_run_writes_every_key_it_owns(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch)
    update = supervisor.run(base_state())

    assert update["instrument"] == "EURUSD"
    assert update["timeframe"] == "H1"
    assert update["language"] == "cs"
    assert update["scope"] == "both"
    assert update["next_agent"] == "analytics"
    assert update["step_count"] == 1
    assert update["messages"]


def test_run_returns_only_keys_this_node_owns(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch)
    update = supervisor.run(base_state())
    assert set(update) <= {
        "instrument",
        "timeframe",
        "language",
        "scope",
        "next_agent",
        "step_count",
        "messages",
        "errors",
    }


def test_run_does_not_mutate_the_state(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch)
    state = base_state(step_count=3, instrument="EURUSD")
    snapshot = json.dumps(state, sort_keys=True, default=str)
    supervisor.run(state)
    assert json.dumps(state, sort_keys=True, default=str) == snapshot


def test_the_trace_line_records_the_models_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch, reason="chart already fetched")
    message = supervisor.run(base_state())["messages"][0]
    assert "chart already fetched" in message.content


# --- what the node refuses to trust ------------------------------------------


def test_the_instrument_is_uppercased_with_its_suffix_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`EURUSD.pro` is the symbol as MT5 knows it; stripping the suffix here would break
    the trader. Only analytics strips it, and only for its search query."""
    stub_decision(monkeypatch, instrument="  eurusd.pro ")
    assert supervisor.run(base_state())["instrument"] == "EURUSD.PRO"


@pytest.mark.parametrize("value", [None, "", "   ", 42, ["EURUSD"]])
def test_an_unusable_instrument_becomes_none(
    value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_decision(monkeypatch, instrument=value)
    assert supervisor.run(base_state())["instrument"] is None


@pytest.mark.parametrize("value", ["everything", "News", "", None, "news chart"])
def test_an_out_of_range_scope_becomes_none_not_a_guess(
    value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structured-output schema constrains a model, it does not bind it."""
    stub_decision(monkeypatch, scope=value)
    assert supervisor.run(base_state())["scope"] is None


@pytest.mark.parametrize("value", ["charting", "", None, "supervisor"])
def test_an_out_of_range_next_agent_becomes_none(
    value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_decision(monkeypatch, next_agent=value)
    assert supervisor.run(base_state())["next_agent"] is None


@pytest.mark.parametrize("value", ["H3", "1h", "", None, "hourly"])
def test_an_unknown_timeframe_falls_back_to_the_default(
    value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_decision(monkeypatch, timeframe=value)
    assert supervisor.run(base_state())["timeframe"] == DEFAULT_TIMEFRAME


def test_a_lowercase_timeframe_is_accepted_and_normalised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_decision(monkeypatch, timeframe="h4")
    assert supervisor.run(base_state())["timeframe"] == "H4"


# --- language is detected once ------------------------------------------------


def test_an_already_detected_language_is_never_revised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4: detected once from the original request. A clarification answered in
    another language must not switch the document's language."""
    stub_decision(monkeypatch, language="en")
    assert supervisor.run(base_state(language="cs"))["language"] == "cs"


def test_language_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch, language=None)
    assert supervisor.run(base_state())["language"] == "en"


# --- rule 3's explanation, and the termination invariants ---------------------


def test_the_clarify_cap_is_explained_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The router returns "done" here but cannot write to state — it is pure — so the node
    has to say why the run ended (spec §5 rule 3)."""
    stub_decision(monkeypatch, instrument=None, scope=None)

    update = supervisor.run(base_state(clarify_count=MAX_CLARIFICATIONS))

    assert any("giving up after" in e for e in update["errors"])
    assert any("instrument" in e and "scope" in e for e in update["errors"])


def test_no_explanation_while_clarifications_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_decision(monkeypatch, instrument=None, scope=None)
    update = supervisor.run(base_state(clarify_count=MAX_CLARIFICATIONS - 1))
    assert "errors" not in update


def test_no_explanation_when_everything_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch)
    update = supervisor.run(base_state(clarify_count=MAX_CLARIFICATIONS))
    assert "errors" not in update


def test_the_step_limit_is_explained_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch)
    update = supervisor.run(base_state(step_count=MAX_STEPS - 1))
    assert update["step_count"] == MAX_STEPS
    assert any("step limit" in e for e in update["errors"])


def test_step_count_increments_even_when_the_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invariant that stops an infinite loop: a supervisor whose model is down must
    still march towards the step limit, or the run dies on GraphRecursionError instead."""

    def boom(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(supervisor, "decide", boom)

    update = supervisor.run(base_state(step_count=4))

    assert update["step_count"] == 5
    assert update["next_agent"] == "done"
    assert any("model unreachable" in e for e in update["errors"])


def test_a_failed_run_proposes_done_and_writes_no_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor, "decide", lambda *a, **k: 1 / 0)
    update = supervisor.run(base_state())
    assert set(update) == {"next_agent", "step_count", "errors"}


def test_repeated_visits_keep_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch)
    state = base_state()
    for expected in (1, 2, 3):
        update = supervisor.run(state)
        assert update["step_count"] == expected
        state["step_count"] = update["step_count"]


# --- the node and the router agree -------------------------------------------


def test_the_node_output_routes_where_the_model_proposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End of the loop: what run() writes is what route_from_supervisor reads."""
    stub_decision(monkeypatch, scope="both", next_agent="trader")

    update = supervisor.run(base_state())
    routed = supervisor.route_from_supervisor({**base_state(), **update})

    assert routed == "trader"


def test_an_unresolved_request_routes_to_clarify(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch, instrument=None, scope=None, next_agent="clarify")

    update = supervisor.run(base_state())
    routed = supervisor.route_from_supervisor({**base_state(), **update})

    assert routed == "clarify"


# --- resolution is monotonic --------------------------------------------------


def test_a_resolved_instrument_is_never_un_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Found on a live run: visit 1 read the request correctly and routed to an agent,
    visit 2 came back unsure, and the router — seeing an unresolved state — sent the user to
    `clarify` mid-run, asking a question the graph already had the answer to.
    """
    stub_decision(monkeypatch, instrument=None, scope=None)

    update = supervisor.run(base_state(instrument="EURUSD", scope="both"))

    assert update["instrument"] == "EURUSD"
    assert update["scope"] == "both"


def test_a_resolved_value_can_still_be_changed_to_another_valid_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sticky, not frozen: a clarification that corrects the symbol has to win."""
    stub_decision(monkeypatch, instrument="XAUUSD", scope="chart")

    update = supervisor.run(base_state(instrument="EURUSD", scope="both"))

    assert update["instrument"] == "XAUUSD"
    assert update["scope"] == "chart"


def test_an_invalid_scope_falls_back_to_the_resolved_one_not_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_decision(monkeypatch, scope="everything")
    assert supervisor.run(base_state(scope="news"))["scope"] == "news"


def test_nothing_resolved_yet_still_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not invent a value on the first visit \u2014 that is what clarify is for."""
    stub_decision(monkeypatch, instrument=None, scope=None)

    update = supervisor.run(base_state())

    assert update["instrument"] is None
    assert update["scope"] is None


def test_a_mid_run_wobble_does_not_burn_a_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user-visible symptom: a spurious question mid-run, and one of the two
    clarifications gone with it."""
    stub_decision(monkeypatch, instrument=None, scope=None)

    update = supervisor.run(base_state(instrument="EURUSD", scope="both", clarify_count=0))
    routed = supervisor.route_from_supervisor(
        {**base_state(instrument="EURUSD", scope="both"), **update}
    )

    assert routed != "clarify"
