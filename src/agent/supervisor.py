"""Supervisor node + the router. Spec §4 `supervisor`, §5 router logic.

The supervisor keeps the decisions that need judgement — what the user meant, and which
artifact to fetch first when both are needed. `route_from_supervisor` keeps the ones that
must not be left to a model: the step limit, the scope guard, and the writer.

The model's answers are all **validated** before they reach state. `scope`,
`next_agent` and `timeframe` are enum-shaped, and an out-of-range value becomes `None`
(or the default) rather than being trusted — a structured-output schema constrains a
model, it does not bind it.

`run()` never raises; on failure it proposes `done` and records why.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, cast

from langchain_core.messages import AIMessage

from src.agent.config import (
    DEFAULT_TIMEFRAME,
    MAX_CLARIFICATIONS,
    MAX_STEPS,
    MT5_TIMEFRAMES,
    chat_model,
)
from src.agent.state import VALID_SCOPES, AgentState, NextAgent, RouteLabel, Scope

AGENT = "supervisor"

ArtifactAgent = Literal["analytics", "trader"]

REQUIRED_ARTIFACTS: dict[Scope, tuple[tuple[ArtifactAgent, str], ...]] = {
    "news": (("analytics", "analytics_result"),),
    "chart": (("trader", "chart"),),
    "both": (("analytics", "analytics_result"), ("trader", "chart")),
}
"""Per spec §5 rule 4: which agent must have run, and the state key that proves it.

Order matters — it is the tie-break when the LLM's proposal is unusable, so `both`
falls back to news-then-chart.
"""

NEXT_AGENTS: tuple[str, ...] = ("clarify", "analytics", "trader", "writer", "done")


# A hand-written JSON Schema rather than a TypedDict: `scope` has to be a nullable enum
# ("news" | "chart" | "both" | null), which a TypedDict annotation cannot express to the
# model as one constrained field.
_DECISION_SCHEMA: dict[str, Any] = {
    "title": "SupervisorDecision",
    "description": "What the supervisor resolved from the request and the conversation.",
    "type": "object",
    "properties": {
        "instrument": {
            "type": ["string", "null"],
            "description": (
                "Broker symbol as it exists on MT5, uppercase, suffix included "
                "(EURUSD, XAUUSD, EURUSD.pro). null if the request does not identify one."
            ),
        },
        "timeframe": {
            "type": "string",
            "enum": list(MT5_TIMEFRAMES),
            "description": "MT5 timeframe.",
        },
        "language": {
            "type": "string",
            "description": "ISO 639-1 code of the ORIGINAL user request.",
        },
        "scope": {
            "type": ["string", "null"],
            "enum": [*VALID_SCOPES, None],
            "description": "What the user asked for. null if genuinely ambiguous.",
        },
        "next_agent": {
            "type": "string",
            "enum": list(NEXT_AGENTS),
            "description": "Proposal for what runs next; the router may override it.",
        },
        "reason": {
            "type": "string",
            "description": "One short sentence explaining the proposal.",
        },
    },
    "required": ["instrument", "timeframe", "language", "scope", "next_agent", "reason"],
}


_PROMPT = """You are the supervisor of a broker-side instrument-analysis graph.

ORIGINAL user request — the only source for `language`:
{request}

Conversation so far, which may contain a clarifying question and its answer:
{conversation}

Already resolved (keep these unless the conversation changes them):
  instrument: {instrument}
  timeframe:  {timeframe}
  scope:      {scope}

Artifacts already produced:
  news analysis: {has_news}
  chart:         {has_chart}
  document:      {has_document}

Your jobs:
1. `instrument` — the broker symbol as it exists on MT5: uppercase, suffix kept.
2. `timeframe` — one of the listed MT5 timeframes. Use {default_timeframe} unless the
   request asks for another.
3. `scope` — "news" if only market news was asked for, "chart" if only the chart and its
   levels, "both" if both.
4. `language` — the ISO 639-1 code of the ORIGINAL request above. Detect it once; it does
   not change later in the run.
5. `next_agent` — what should run next.

Rules:
- **Guessing is worse than asking.** Use null for `instrument` or `scope` when the request
  is genuinely ambiguous. One clarifying question costs a turn; a wrong instrument
  produces a confidently wrong report.
- If either is null, propose "clarify".
- Otherwise propose the agent whose artifact is still missing: "analytics" for the news
  analysis, "trader" for the chart. When both are missing, pick the one to do first.
- Propose "writer" only once every artifact the scope requires exists.
- Propose "done" only once the document exists.
- Do not propose an agent outside the scope that was asked for."""


def _conversation(state: AgentState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return "- nothing yet"
    return "\n".join(
        f"- {type(m).__name__}: {getattr(m, 'content', m)}" for m in messages
    )


def _normalise_instrument(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    symbol = value.strip().upper()
    return symbol or None


def _normalise_scope(value: Any) -> Scope | None:
    return value if value in VALID_SCOPES else None


def _normalise_next_agent(value: Any) -> NextAgent | None:
    return value if value in NEXT_AGENTS else None


def _normalise_timeframe(value: Any) -> str:
    if isinstance(value, str) and value.strip().upper() in MT5_TIMEFRAMES:
        return value.strip().upper()
    return DEFAULT_TIMEFRAME


def _runtime_config() -> Mapping[str, Any] | None:
    """The invocation's `configurable`, for the per-agent model override of spec §6.

    Read via `langgraph.config.get_config()` rather than a second node parameter, so the
    entry point keeps the `(state) -> dict` shape §2 requires.
    """
    try:
        from langgraph.config import get_config

        return get_config()
    except Exception:
        return None


def decide(state: AgentState, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One structured-output call. Returns the model's raw decision dict."""
    prompt = _PROMPT.format(
        request=state.get("request") or "(empty)",
        conversation=_conversation(state),
        instrument=state.get("instrument") or "not resolved",
        timeframe=state.get("timeframe") or DEFAULT_TIMEFRAME,
        scope=state.get("scope") or "not resolved",
        has_news="yes" if state.get("analytics_result") is not None else "no",
        has_chart="yes" if state.get("chart") is not None else "no",
        has_document="yes" if state.get("document") is not None else "no",
        default_timeframe=DEFAULT_TIMEFRAME,
    )
    model = chat_model(AGENT, config).with_structured_output(_DECISION_SCHEMA)
    decision = model.invoke(prompt)
    if not isinstance(decision, dict):
        raise ValueError(f"model returned {type(decision).__name__}, expected a decision dict")
    return decision


def run(state: AgentState) -> dict[str, Any]:
    """Resolve instrument/timeframe/scope/language and propose the next agent.

    Also owns the `errors` entry demanded by spec §5 rule 3: when `instrument` or `scope`
    is still unresolved and `clarify_count` has hit `MAX_CLARIFICATIONS`, the run ends and
    something has to say why. The router cannot write it — it is pure — so it happens
    here.
    """
    step_count = state.get("step_count", 0) + 1

    try:
        decision = decide(state, _runtime_config())

        # Resolution is **monotonic**: once the request has yielded an instrument and a
        # scope, a later visit may *change* them to another valid value but can never
        # un-resolve them back to null. Without this the graph asks a question it already
        # had the answer to: visit 1 reads "eurodolar ... graf" correctly and routes to an
        # agent, visit 2 comes back unsure, and the router — seeing an unresolved state —
        # sends the user to `clarify` mid-run. The prompt asks the model to keep them; this
        # is the guarantee, because the prompt is a request and this is not.
        instrument = _normalise_instrument(decision.get("instrument")) or state.get("instrument")
        scope = _normalise_scope(decision.get("scope")) or state.get("scope")

        next_agent = _normalise_next_agent(decision.get("next_agent"))
        timeframe = _normalise_timeframe(decision.get("timeframe"))

        # Detected once from the original request and not revised (spec §4).
        language = state.get("language") or decision.get("language") or "en"

        update: dict[str, Any] = {
            "instrument": instrument,
            "timeframe": timeframe,
            "language": language,
            "scope": scope,
            "next_agent": next_agent,
            "step_count": step_count,
            "messages": [
                AIMessage(
                    content=(
                        f"supervisor: instrument={instrument} scope={scope} "
                        f"next={next_agent} — {decision.get('reason') or 'no reason given'}"
                    )
                )
            ],
        }

        # Rule 3's explanation. The router will return "done" on this state; without this
        # entry the run would end silently and nobody could tell why.
        unresolved = not instrument or scope is None
        if unresolved and state.get("clarify_count", 0) >= MAX_CLARIFICATIONS:
            missing = ", ".join(
                name
                for name, value in (("instrument", instrument), ("scope", scope))
                if not value
            )
            update["errors"] = [
                f"{AGENT}: giving up after {MAX_CLARIFICATIONS} clarifications; "
                f"still unresolved: {missing}"
            ]
        elif step_count >= MAX_STEPS:
            update["errors"] = [
                f"{AGENT}: step limit {MAX_STEPS} reached without a document"
            ]

        return update

    except Exception as exc:  # noqa: BLE001 - the node contract is "never raise"
        return {
            "next_agent": "done",
            "step_count": step_count,
            "errors": [f"{AGENT}: {exc!r}"],
        }


def _cli() -> None:
    """Resolve one request and show both the decision and where it would route.

    The most useful of the CLIs when iterating on the prompt: it shows the model's raw
    proposal, the validated update, and the router's verdict side by side.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the supervisor on one request.")
    parser.add_argument("request")
    parser.add_argument("--prompt-only", action="store_true", help="print the prompt, no call")
    args = parser.parse_args()

    state = AgentState(request=args.request)

    if args.prompt_only:
        print(
            _PROMPT.format(
                request=args.request,
                conversation=_conversation(state),
                instrument="not resolved",
                timeframe=DEFAULT_TIMEFRAME,
                scope="not resolved",
                has_news="no",
                has_chart="no",
                has_document="no",
                default_timeframe=DEFAULT_TIMEFRAME,
            )
        )
        return

    update = run(state)
    print(json.dumps(update, indent=2, default=str, ensure_ascii=False))

    # `**` expansion into a TypedDict is not expressible, hence the cast.
    merged = cast(AgentState, {**state, **update})
    print("\nrouter ->", route_from_supervisor(merged))


def route_from_supervisor(state: AgentState) -> RouteLabel:
    """Pure function of state. No I/O, no LLM, no clock, no randomness.

    Purity is not cosmetic: an impure router makes the graph unreplayable from a
    checkpoint, which silently breaks resume after `clarify`.

    The five rules of spec §5, in order. The order is load-bearing — the step check must
    come first (otherwise a runaway loop never terminates) and the scope check must
    override the LLM's proposal (otherwise the supervisor can run an agent the user never
    asked for, or finish with no document).
    """
    # 1. Runaway guard, before anything else. Fires ahead of RECURSION_LIMIT so the run
    #    ends with a state we can inspect rather than a GraphRecursionError.
    if state.get("step_count", 0) >= MAX_STEPS:
        return "step_limit"

    # 2. The document is the only definition of finished.
    if state.get("document") is not None:
        return "done"

    # 3. Nothing can be fetched until we know what and about which symbol. `scope` is
    #    re-validated against VALID_SCOPES rather than trusted: it arrives from the LLM,
    #    so the annotation is a hope, not a guarantee.
    scope = state.get("scope")
    if not state.get("instrument") or scope is None or scope not in VALID_SCOPES:
        if state.get("clarify_count", 0) < MAX_CLARIFICATIONS:
            return "clarify"
        # Asking a third time is harassment. `run()` records why in `errors`.
        return "done"

    # 4. Only the agents this scope requires, and only the ones that have not produced
    #    their artifact yet.
    missing: list[ArtifactAgent] = [
        agent for agent, key in REQUIRED_ARTIFACTS[scope] if state.get(key) is None
    ]
    if missing:
        # The LLM keeps the judgement call — which to fetch first when both are needed —
        # but only within `missing`. Anything else is overridden.
        proposal = state.get("next_agent")
        for agent in missing:
            if proposal == agent:
                return agent
        return missing[0]

    # 5. Inputs complete: write the document. The writer cannot be skipped.
    return "writer"


if __name__ == "__main__":
    _cli()
