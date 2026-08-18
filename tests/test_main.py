"""`src/main.py` — the caller. Boundaries stubbed, the graph itself runs for real.

What is worth testing here is not the analysis but the *caller's* three jobs: the
thread_id, the checkpointer, and the clarification loop. Plus the one failure mode a CLI
must never have — hanging on `input()` when nobody is there to type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src import main
from src.agent import supervisor
from src.agents import analytics, trader, writer
from tests.ohlc_fixture import EURUSD

SECTIONS: dict[str, Any] = {
    "title": "EURUSD — analýza",
    "news_heading": "Zprávy",
    "news_analysis": "Sazby zůstaly (https://example.com/a).",
    "chart_heading": "Graf",
    "chart_analysis": "Cena uprostřed rozpětí.",
    "levels_heading": "Úrovně",
    "level_columns": ["Typ", "Cena", "Zóna", "Testy", "Vstup"],
    "sources_heading": "Zdroje",
}


@pytest.fixture(autouse=True)
def boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub MT5, Tavily and the models; keep every file inside tmp_path."""
    monkeypatch.setattr(trader, "CHART_DIR", str(tmp_path / "charts"))
    monkeypatch.setattr(writer, "REPORT_DIR", str(tmp_path / "reports"))

    monkeypatch.setattr(trader, "fetch_ohlc", lambda *a, **k: list(EURUSD))
    monkeypatch.setattr(trader, "interpret_zones", lambda *a, **k: "mid-range")
    monkeypatch.setattr(
        analytics,
        "search_news",
        lambda *a, **k: [
            {"title": "T", "url": "https://example.com/a", "published": "", "snippet": ""}
        ],
    )
    monkeypatch.setattr(analytics, "summarise", lambda *a, **k: "S (https://example.com/a).")
    monkeypatch.setattr(writer, "build_sections", lambda **k: (dict(SECTIONS), []))

    # Not a tty under pytest, so the CLI is non-interactive unless a test says otherwise.
    monkeypatch.setattr(supervisor, "decide", resolved())


def resolved(scope: str = "news") -> Any:
    def decide(state: Any, config: Any = None) -> dict[str, Any]:
        return {
            "instrument": "EURUSD",
            "timeframe": "H1",
            "language": "cs",
            "scope": scope,
            "next_agent": "analytics",
            "reason": "stub",
        }

    return decide


def needs_clarifying(scope: str = "news") -> Any:
    """Ambiguous until a human answer lands in `messages`."""
    from langchain_core.messages import HumanMessage

    def decide(state: Any, config: Any = None) -> dict[str, Any]:
        answered = any(isinstance(m, HumanMessage) for m in (state.get("messages") or []))
        return {
            "instrument": "EURUSD" if answered else None,
            "timeframe": "H1",
            "language": "cs",
            "scope": scope if answered else None,
            "next_agent": "analytics" if answered else "clarify",
            "reason": "stub",
        }

    return decide


# --- happy path ---------------------------------------------------------------


def test_a_resolvable_request_exits_zero_and_names_the_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main.main(["co je nového na eurodolaru", "--thread-id", "t1"])

    assert code == main.EXIT_OK
    out = capsys.readouterr().out
    assert "Document:" in out
    assert SECTIONS["title"] in out

    path = next(line.split("Document:")[1].strip() for line in out.splitlines() if "Document:" in line)
    assert Path(path).is_file()


def test_the_chart_is_reported_when_the_scope_produced_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(supervisor, "decide", resolved(scope="both"))

    main.main(["zprávy i graf", "--thread-id", "t-chart"])

    assert "Chart:" in capsys.readouterr().out


def test_json_output_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    main.main(["zprávy", "--thread-id", "t-json", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["document"]["format"] == "docx"


def test_progress_goes_to_stderr_so_stdout_stays_parseable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--json` output must be pipeable; progress lines would corrupt it."""
    main.main(["zprávy", "--thread-id", "t-streams", "--json"])

    captured = capsys.readouterr()
    json.loads(captured.out)  # would raise if progress leaked into stdout
    assert "analytics" in captured.err


def test_quiet_suppresses_progress(capsys: pytest.CaptureFixture[str]) -> None:
    main.main(["zprávy", "--thread-id", "t-quiet", "--quiet"])
    assert capsys.readouterr().err == ""


# --- the thread_id ------------------------------------------------------------


def test_a_thread_id_is_generated_when_none_is_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§6 says the graph must not invent one; the caller may, and this is the caller."""
    main.main(["zprávy"])
    assert "thread_id: cli-" in capsys.readouterr().err


def test_a_supplied_thread_id_is_used_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    main.main(["zprávy", "--thread-id", "req-4711"])
    assert "thread_id: req-4711" in capsys.readouterr().err


def test_generated_thread_ids_differ_between_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main.main(["zprávy"])
    first = capsys.readouterr().err
    main.main(["zprávy"])
    second = capsys.readouterr().err
    assert first != second


# --- clarifications -----------------------------------------------------------


def test_a_scripted_answer_completes_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(supervisor, "decide", needs_clarifying())

    code = main.main(["něco", "--thread-id", "t-answer", "--answer", "EURUSD, jen zprávy"])

    out = capsys.readouterr().out
    assert code == main.EXIT_OK
    assert "?" in out  # the question was shown
    assert "Document:" in out


def test_answers_are_consumed_in_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two clarifications are possible, so the loop has to keep asking, not ask once."""
    monkeypatch.setattr(supervisor, "decide", needs_clarifying())

    code = main.main(
        ["něco", "--thread-id", "t-order", "--answer", "nevím", "--answer", "EURUSD"]
    )

    assert code == main.EXIT_OK
    assert capsys.readouterr().out.count("?") >= 1


def test_running_out_of_answers_suspends_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure a CLI must never have: blocking on `input()` with nobody there. stdin is
    not a tty under pytest, so the run reports and exits instead."""
    monkeypatch.setattr(supervisor, "decide", needs_clarifying())

    code = main.main(["něco", "--thread-id", "t-suspend"])

    captured = capsys.readouterr()
    assert code == main.EXIT_NEEDS_ANSWER
    assert "?" in captured.out
    assert "suspended" in captured.err
    assert "No document produced." in captured.out


def test_a_suspended_thread_resumes_on_the_same_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of exposing `--thread-id`: an answer can arrive later.

    Same *process* here, because the default InMemorySaver lives in memory — that is
    exactly why `--postgres` exists.
    """
    monkeypatch.setattr(supervisor, "decide", needs_clarifying())

    with main.compiled_graph(use_postgres=False) as graph:
        config = {"configurable": {"thread_id": "t-resume"}, "recursion_limit": 40}

        state, code = main.drive(graph, "něco", config, interactive=False, quiet=True)
        assert code == main.EXIT_NEEDS_ANSWER
        assert state.get("document") is None

        state, code = main.drive(
            graph, "něco", config, answers=["EURUSD, jen zprávy"], interactive=False, quiet=True
        )
        assert code == main.EXIT_OK
        assert state["document"] is not None


def test_an_interactive_prompt_is_used_when_stdin_is_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda *a: asked.append("EURUSD") or "EURUSD")

    answer = main.next_answer([], "Který instrument?", interactive=True)

    assert answer == "EURUSD"
    assert asked == ["EURUSD"]


def test_eof_on_stdin_is_treated_as_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_eof(*_: Any) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert main.next_answer([], "Který instrument?", interactive=True) is None


# --- failure reporting --------------------------------------------------------


def test_a_run_without_a_document_exits_one_and_lists_the_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        writer, "build_sections", lambda **k: (_ for _ in ()).throw(RuntimeError("no key"))
    )

    code = main.main(["zprávy", "--thread-id", "t-fail"])

    out = capsys.readouterr().out
    assert code == main.EXIT_NO_DOCUMENT
    assert "No document produced." in out
    assert "no key" in out


def test_a_degraded_run_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A document that reports a gap is a success: exit 0 with the errors listed."""
    monkeypatch.setattr(supervisor, "decide", resolved(scope="both"))
    monkeypatch.setattr(
        trader, "fetch_ohlc", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("MT5 down"))
    )

    code = main.main(["zprávy i graf", "--thread-id", "t-degraded"])

    out = capsys.readouterr().out
    assert code == main.EXIT_OK
    assert "Document:" in out
    assert "MT5 down" in out
    assert "Chart:" not in out  # the sentinel has no path


# --- the checkpointer ---------------------------------------------------------


def test_postgres_without_a_connection_string_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_URI", raising=False)

    with pytest.raises(SystemExit, match="POSTGRES_URI"):
        with main.compiled_graph(use_postgres=True):
            pass


def test_the_default_checkpointer_needs_no_database() -> None:
    with main.compiled_graph(use_postgres=False) as graph:
        assert graph is not None


# --- argument parsing --------------------------------------------------------


def test_the_request_is_required() -> None:
    with pytest.raises(SystemExit):
        main.build_parser().parse_args([])


def test_answers_accumulate() -> None:
    args = main.build_parser().parse_args(["r", "--answer", "a", "--answer", "b"])
    assert args.answer == ["a", "b"]


# --- the trace ----------------------------------------------------------------


def test_the_trace_shows_each_step_and_what_it_wrote(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reconstructed by diffing checkpoints: langgraph 1.2.11 keeps no per-node `writes` in
    checkpoint metadata, but it keeps the full state at every step."""
    monkeypatch.setattr(supervisor, "decide", resolved(scope="news"))

    main.main(["zprávy", "--thread-id", "t-trace", "--trace", "--quiet"])

    out = capsys.readouterr().out
    assert "--- trace ---" in out
    assert "next: analytics" in out
    assert "next: writer" in out
    assert "next: end" in out
    assert "analytics_result:" in out
    assert "document:" in out


def test_the_trace_is_oldest_first() -> None:
    """`get_state_history` yields newest first; reading a run backwards is useless."""
    with main.compiled_graph(use_postgres=False) as graph:
        config = {"configurable": {"thread_id": "t-order"}, "recursion_limit": 40}
        main.drive(graph, "zprávy", config, interactive=False, quiet=True)

        steps = [s.metadata.get("step") for s in graph.get_state_history(config)][::-1]
        assert steps == sorted(steps)


def test_no_trace_unless_asked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main.main(["zprávy", "--thread-id", "t-no-trace", "--quiet"])
    assert "--- trace ---" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "value,expected",
    [
        ([], "[]"),
        ("short", "short"),
        ("x" * 200, "x" * 87 + "…"),
        ({"a": 1, "b": 2}, "{a=1, b=2}"),
        ({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, "{a=1, b=2, c=3, d=4…}"),
    ],
    ids=["empty-list", "short-string", "long-string", "small-dict", "truncated-dict"],
)
def test_state_values_are_summarised_to_one_line(value: Any, expected: str) -> None:
    """A trace line that wraps is a trace nobody reads."""
    assert main._short(value) == expected


def test_a_message_list_is_summarised_by_its_last_content() -> None:
    from langchain_core.messages import AIMessage

    rendered = main._short([AIMessage(content="first"), AIMessage(content="second")])

    assert rendered.startswith("[2] last: second")
