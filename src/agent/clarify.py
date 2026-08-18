"""Human-in-the-loop interrupt node. Spec §4 `clarify`.

Asks exactly one question about whichever of `instrument` / `scope` is unresolved, records
the answer, and hands control back. **It decides nothing** — the supervisor re-parses
`messages` on the next visit.

Two things worth knowing before editing this file:

* **`interrupt()` works by raising.** `GraphInterrupt` subclasses `GraphBubbleUp`, the
  marker LangGraph uses for control-flow signals, and it *must* propagate. A blanket
  `except Exception` here would swallow the suspend and turn the whole human-in-the-loop
  mechanism into a silent no-op. This node is therefore the one place the "agents never
  raise" rule does not apply, and it has no try/except at all.
* **The node re-executes on resume.** Everything before `interrupt()` runs twice, so
  `build_payload` is pure and the counter is derived from state rather than incremented
  in place.

Spec §4 gives this node no model, so the question comes from a template table rather than
an LLM. `QUESTION_TEMPLATES` covers Czech and English and falls back to English for
anything else.
"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agent.state import VALID_SCOPES, AgentState

AGENT = "clarify"

FALLBACK_LANGUAGE = "en"

QUESTION_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "instrument": "Which instrument should I analyse? (e.g. EURUSD, XAUUSD)",
        "scope": "Should I look at market news, the chart, or both?",
        "both": (
            "Which instrument should I analyse, and should I look at market news, "
            "the chart, or both?"
        ),
    },
    "cs": {
        "instrument": "Který instrument mám analyzovat? (např. EURUSD, XAUUSD)",
        "scope": "Mám se podívat na zprávy, na graf, nebo na obojí?",
        "both": (
            "Který instrument mám analyzovat a mám se podívat na zprávy, na graf, "
            "nebo na obojí?"
        ),
    },
}


# Characters that appear in Czech and in essentially nothing else this vault sees. Used only
# as a last resort — see `language_for`.
_CZECH_MARKERS = frozenset("ěščřžýáíéúůďťňĚŠČŘŽÝÁÍÉÚŮĎŤŇ")


def language_for(state: AgentState) -> str:
    """The language to ask in: what the supervisor detected, else a guess, else English.

    The supervisor normally owns this. But on the *first* visit its model call can fail, and
    then it writes no `language` at all — which used to mean a Czech request got an English
    question, because this node has no model of its own to fall back on.

    The guess is a diacritics probe, not language detection: it recognises Czech and nothing
    else, and it only ever runs when `language` is still unset. A Czech request typed without
    diacritics still gets English, which is the honest limit of a table with two languages in
    it.
    """
    detected = state.get("language")
    if detected:
        return detected
    if _CZECH_MARKERS & set(state.get("request") or ""):
        return "cs"
    return FALLBACK_LANGUAGE


def missing_fields(state: AgentState) -> list[str]:
    """Which of `instrument` / `scope` still needs an answer.

    Same predicate the router uses in rule 3, so the two cannot disagree about whether a
    clarification is warranted.
    """
    missing: list[str] = []
    if not state.get("instrument"):
        missing.append("instrument")
    if state.get("scope") not in VALID_SCOPES:
        missing.append("scope")
    return missing


def question_for(missing: list[str], language: str) -> str:
    """One question, in `language`. Exactly one, even when both fields are unresolved."""
    templates = QUESTION_TEMPLATES.get(language, QUESTION_TEMPLATES[FALLBACK_LANGUAGE])
    if len(missing) > 1:
        return templates["both"]
    if missing:
        return templates[missing[0]]
    # Reached only if the router sends us here with nothing missing; ask the broader
    # question rather than interrupting with an empty prompt.
    return templates["both"]


def build_payload(state: AgentState) -> dict[str, Any]:
    """The interrupt payload. Pure — the node re-executes this on resume.

    Carries what was already understood alongside the question, so the caller can render
    the state of the conversation rather than just a bare prompt (spec §4).
    """
    language = language_for(state)
    missing = missing_fields(state)
    return {
        "agent": AGENT,
        "question": question_for(missing, language),
        "language": language,
        "missing": missing,
        "understood": {
            "instrument": state.get("instrument"),
            "timeframe": state.get("timeframe"),
            "scope": state.get("scope"),
        },
        "request": state.get("request"),
        "attempt": state.get("clarify_count", 0) + 1,
    }


def _answer_text(resumed: Any) -> str:
    """The answer out of whatever the caller resumed with.

    Spec §4 documents `Command(resume={"answer": ...})`, but a bare string is the obvious
    mistake to make and costs nothing to accept.
    """
    if isinstance(resumed, Mapping):
        value = resumed.get("answer")
    else:
        value = resumed
    return "" if value is None else str(value).strip()


def run(state: AgentState) -> dict[str, Any]:
    """Ask, record, hand back. Writes `clarify_count` and `messages` only.

    An empty answer still consumes one clarification: the supervisor will find nothing new
    in `messages`, and rule 3 eventually ends the run rather than asking forever.
    """
    payload = build_payload(state)

    # DO NOT wrap this in try/except. `interrupt()` suspends the run by raising
    # GraphInterrupt (a GraphBubbleUp); swallowing it would turn human-in-the-loop into a
    # silent no-op that returns a clarification nobody was ever asked. This node is the
    # one place the "agents never raise" rule does not apply.
    resumed = interrupt(payload)

    answer = _answer_text(resumed)

    messages: list[Any] = [AIMessage(content=payload["question"])]
    if answer:
        messages.append(HumanMessage(content=answer))

    return {
        "clarify_count": state.get("clarify_count", 0) + 1,
        "messages": messages,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Print the interrupt payload this node would send. Does not interrupt."
    )
    parser.add_argument("--request", default="")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--language", default=FALLBACK_LANGUAGE)
    args = parser.parse_args()

    minimal = AgentState(
        request=args.request,
        instrument=args.instrument,
        scope=args.scope,
        language=args.language,
    )
    print(json.dumps(build_payload(minimal), indent=2, ensure_ascii=False))
