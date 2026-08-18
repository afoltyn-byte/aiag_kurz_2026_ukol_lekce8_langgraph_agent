"""CLI entry point: one request in, one document out.

This is the **caller**, not part of the graph. It owns the three things spec §6 assigns to
whoever invokes the graph rather than to the graph itself:

* the `thread_id` — one per request, generated here when not supplied;
* the checkpointer — `InMemorySaver` by default, `PostgresSaver` with `--postgres`;
* the clarification loop — `interrupt()` hands control back to the caller, and answering
  is the caller's job.

No routing, no prompts, no state shaping. Everything it prints comes out of state.

Exit codes: `0` a document was produced, `1` the run finished without one, `2` the run
needs an answer and none was available.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Sequence
from uuid import uuid4

from langgraph.types import Command

from src.agent.config import RECURSION_LIMIT, postgres_uri
from src.agent.graph import compile_graph

EXIT_OK = 0
EXIT_NO_DOCUMENT = 1
EXIT_NEEDS_ANSWER = 2


@contextmanager
def compiled_graph(*, use_postgres: bool) -> Iterator[Any]:
    """The graph, with the checkpointer the caller asked for.

    A context manager because `PostgresSaver.from_conn_string` is one: the connection has
    to outlive the whole run, so the graph must be used inside that block rather than
    handed back from a factory.
    """
    if not use_postgres:
        # InMemorySaver: enough for a single-process run, and a resume after `clarify`
        # only works while this process lives.
        yield compile_graph()
        return

    uri = postgres_uri()
    if not uri:
        raise SystemExit("--postgres needs POSTGRES_URI in the environment")

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(uri) as saver:
        saver.setup()  # idempotent; creates the checkpoint tables on first use
        yield compile_graph(checkpointer=saver)


def _progress(node: str, update: Any) -> str:
    """One line per finished node, with the agent's own log entry when it has one."""
    entries = (update or {}).get("agent_log") if isinstance(update, dict) else None
    if entries:
        entry = entries[-1]
        return f"  · {node}: {entry.get('status')} ({entry.get('duration_s')}s)"
    return f"  · {node}"


def run_step(
    graph: Any, payload: Any, config: dict[str, Any], *, quiet: bool
) -> list[dict[str, Any]]:
    """Advance the graph until it finishes or interrupts. Returns interrupt payloads."""
    interrupts: list[dict[str, Any]] = []
    for chunk in graph.stream(payload, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupts.extend(item.value for item in update)
            elif not quiet:
                print(_progress(node, update), file=sys.stderr)
    return interrupts


def next_answer(pending: list[str], question: str, *, interactive: bool) -> str | None:
    """The next scripted answer, else a prompt, else `None` when there is no way to ask."""
    if pending:
        return pending.pop(0)
    if not interactive:
        return None
    try:
        return input("> ")
    except EOFError:
        return None


def drive(
    graph: Any,
    request: str,
    config: dict[str, Any],
    *,
    answers: Sequence[str] = (),
    interactive: bool = True,
    quiet: bool = False,
) -> tuple[dict[str, Any], int]:
    """Run to completion, answering clarifications. Returns (final state, exit code).

    A loop, not a single question: the graph may ask up to `MAX_CLARIFICATIONS` times.
    """
    pending = list(answers)
    payload: Any = {"request": request}

    while True:
        interrupts = run_step(graph, payload, config, quiet=quiet)
        if not interrupts:
            break

        question = interrupts[0].get("question", "(no question in the payload)")
        print(f"\n? {question}")

        answer = next_answer(pending, question, interactive=interactive)
        if answer is None:
            # Nothing to answer with. Leaving the thread suspended is the point: the same
            # thread_id can be resumed later once an answer exists.
            print("No answer available; the run is suspended.", file=sys.stderr)
            return graph.get_state(config).values, EXIT_NEEDS_ANSWER
        payload = Command(resume={"answer": answer})

    state = graph.get_state(config).values
    return state, EXIT_OK if state.get("document") else EXIT_NO_DOCUMENT


def report(state: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(state, indent=2, default=str, ensure_ascii=False))
        return

    document = state.get("document")
    print()
    if document:
        print(f"Document: {document['path']}")
        print(f"  title:  {document['title']}")
        print(f"  scope:  {document['scope']}   language: {document['language']}")
    else:
        print("No document produced.")

    chart = state.get("chart")
    if chart and chart.get("path"):
        print(f"Chart:    {chart['path']}")

    for error in state.get("errors") or []:
        print(f"  ! {error}")


def _short(value: Any, width: int = 88) -> str:
    """One readable line out of anything that lives in state."""
    if isinstance(value, list):
        if not value:
            return "[]"
        tail = getattr(value[-1], "content", value[-1])
        return f"[{len(value)}] last: {_short(tail, width - 20)}"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}={_short(v, 24)}" for k, v in list(value.items())[:4])
        return "{" + inner + ("…}" if len(value) > 4 else "}")
    text = str(value).replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def print_trace(graph: Any, config: dict[str, Any]) -> None:
    """Per-superstep trace: what each step added to state.

    Reconstructed from the checkpointer's own history, oldest first, by diffing consecutive
    snapshots. langgraph 1.2.11 does not record per-node writes in checkpoint metadata, but
    it does keep the full state at every step, and the delta is the same information.

    Only reaches back as far as the checkpointer holds: with the default `InMemorySaver`
    that means this process only, which is what `--postgres` is for.
    """
    snapshots = list(graph.get_state_history(config))[::-1]
    print("\n--- trace ---")

    previous: dict[str, Any] = {}
    for snapshot in snapshots:
        values = dict(snapshot.values)
        changed = {k: v for k, v in values.items() if previous.get(k, object()) != v}
        step = snapshot.metadata.get("step")
        following = ", ".join(snapshot.next) if snapshot.next else "end"

        print(f"step {step:>3}   next: {following}")
        for key in sorted(changed):
            print(f"          {key}: {_short(changed[key])}")
        previous = values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Run one instrument-analysis request end to end.",
    )
    parser.add_argument("request", help="what to analyse, in any language")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="one per request; generated when omitted. Reuse it to resume a suspended run.",
    )
    parser.add_argument(
        "--answer",
        action="append",
        default=[],
        metavar="TEXT",
        help="answer a clarification without a prompt; repeatable, consumed in order",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="use PostgresSaver from POSTGRES_URI, so a suspended run survives a restart",
    )
    parser.add_argument("--json", action="store_true", help="dump the final state as JSON")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="per-superstep trace of what each node wrote, from the checkpointer",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-node progress")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    thread_id = args.thread_id or f"cli-{uuid4().hex[:12]}"
    config = {
        "configurable": {"thread_id": thread_id},
        # Passed explicitly so `step_limit` fires before GraphRecursionError (§6).
        "recursion_limit": RECURSION_LIMIT,
    }

    if not args.quiet:
        print(f"thread_id: {thread_id}", file=sys.stderr)

    with compiled_graph(use_postgres=args.postgres) as graph:
        state, code = drive(
            graph,
            args.request,
            config,
            answers=args.answer,
            # Only prompt when there is a human on the other end; otherwise a scripted or
            # scheduled run would hang forever on `input()`.
            interactive=sys.stdin.isatty(),
            quiet=args.quiet,
        )
        report(state, as_json=args.json)
        # Inside the block: with --postgres the connection has to be open to read history.
        if args.trace:
            print_trace(graph, config)

    return code


if __name__ == "__main__":
    sys.exit(main())
