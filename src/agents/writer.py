"""writer agent — spec §4 `writer`.

Turns the artifacts into one Word document. Reads `analytics_result`, `chart`,
`instrument`, `timeframe`, `scope`, `language`; writes `document`, `errors`, `agent_log`
and a short trace line to `messages`.

Division of labour, per CLAUDE.md non-negotiable 6:

* the **model** writes prose and heading labels, in `state["language"]`;
* **`docbuilder.py`** decides structure and builds the file, with no model involved.

So the model cannot add or drop a section by wording its output differently — `scope`
alone decides that.

`run()` never raises. On failure `document` is `None` plus an `errors` entry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from langchain_core.messages import AIMessage

from src.agent.config import (
    DEFAULT_TIMEFRAME,
    REPORT_DIR,
    chat_model,
)
from src.agent.state import VALID_SCOPES, AgentState
from src.agents.docbuilder import (
    LEVEL_COLUMN_COUNT,
    LEVEL_COLUMNS_FALLBACK,
    DocumentSections,
    build_document,
)

AGENT = "writer"


def report_path(symbol: str, *, now: datetime | None = None) -> str:
    """`{REPORT_DIR}/{symbol}_{YYYYMMDD-HHMMSS}.docx`.

    Timestamped for the same reason the chart is: a retry produces a new file instead of
    corrupting the one a previous attempt half-wrote.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    return str(Path(REPORT_DIR) / f"{symbol}_{stamp}.docx")


_SECTIONS_PROMPT = """You are writing a broker-side analytical report on one instrument.

Instrument: {instrument}
Timeframe: {timeframe}
Requested scope: {scope}
Write EVERYTHING — title, every heading label, and every paragraph — in the language
with this ISO 639-1 code: {language}

News summary retrieved for this instrument (may be empty):
{news_summary}

Sources retrieved (may be empty):
{sources}

Chart status: {chart_status}

Chart levels already computed from the OHLC data (do not recompute or adjust them):
{levels}

The chart analyst's note on those levels, in English — rewrite it in the target
language, do not quote it verbatim (may be empty):
{commentary}

Fill every field of the schema, even the ones this scope will not use.

Rules:
- Base factual claims only on the material above. Add no events, figures or context of
  your own. If a section has no material, say so plainly in that section's paragraph.
- In the news paragraph, follow each claim with the URL it came from, in parentheses.
- `level_columns` must be exactly {column_count} short column labels, in this order:
  level kind, price, zone range, number of touches, entry candidate.
- This is analytical material for a human reader. Give no trading advice, no
  recommendation and no price target of your own.
- Do not translate technical identifiers: the instrument symbol, the timeframe, and
  level kinds such as `resistance` or `session_high` stay as they are."""


def _render_sources(items: Any) -> str:
    if not items:
        return "- none retrieved"
    return "\n".join(
        f"- {item.get('title') or item.get('url')} ({item.get('published') or 'date unknown'}) "
        f"{item.get('url')}"
        for item in items
    )


def _chart_status(chart: Mapping[str, Any] | None) -> str:
    """What the model needs to know to write an honest chart paragraph.

    On the trader's failure path the artifact is present but carries `error` and no image,
    and the paragraph has to say so — in the target language, which is why this is the
    model's job and not a fixed string in `docbuilder`.
    """
    if chart is None:
        return "no chart was requested for this scope"
    if chart.get("error") or not chart.get("path"):
        return (
            "THE CHART COULD NOT BE GENERATED. State this plainly in the chart paragraph "
            "and do not describe any price action or levels. Reason: "
            f"{chart.get('error') or 'unknown'}"
        )
    return "generated successfully"


def _render_levels(levels: Any) -> str:
    if not levels:
        return "- none"
    return "\n".join(
        f"- {lv.get('kind')} {lv.get('price')} (zone {lv.get('low')}-{lv.get('high')}, "
        f"touches {lv.get('touches')}, strength {lv.get('strength')}"
        + (f", entry candidate {lv['entry']}" if lv.get("entry") else "")
        + ")"
        for lv in levels
    )


# Fallbacks for fields the model omits or leaves blank. English on purpose: `docbuilder` is
# language-agnostic and this module cannot translate without another model call. An English
# heading over correct content beats losing the section — and beats a KeyError.
HEADING_FALLBACKS: dict[str, str] = {
    "news_heading": "News analysis",
    "chart_heading": "Chart",
    "levels_heading": "Levels",
    "sources_heading": "Sources",
}

# Prose may legitimately be empty — `docbuilder` skips an empty chart paragraph — so these
# are only ever defaulted from *missing* to `""`, never treated as an error to hide.
PROSE_KEYS = ("news_analysis", "chart_analysis")


def complete_sections(
    raw: Mapping[str, Any], *, instrument: str, timeframe: str
) -> tuple[DocumentSections, list[str]]:
    """Fill anything the model left out. Returns the sections and the keys it had to fix.

    Structured output constrains a model; it does not bind it. A live run came back without
    `sources_heading`, which reached `docbuilder` as a `KeyError` — the router retried the
    writer and the second attempt succeeded, so the run survived, but at the cost of a step
    and eight seconds. `level_columns` was already defended this way; the rest was not, and
    that inconsistency was the bug.

    The returned key list is not cosmetic: a model that keeps omitting fields is worth
    knowing about, so the caller records it in `errors` while still shipping the document.
    """
    sections: dict[str, Any] = dict(raw)
    filled: list[str] = []

    if not str(sections.get("title") or "").strip():
        sections["title"] = f"{instrument} {timeframe}"
        filled.append("title")

    for key, fallback in HEADING_FALLBACKS.items():
        if not str(sections.get(key) or "").strip():
            sections[key] = fallback
            filled.append(key)

    for key in PROSE_KEYS:
        if sections.get(key) is None:
            sections[key] = ""
            filled.append(key)

    columns = sections.get("level_columns")
    if not isinstance(columns, list) or len(columns) != LEVEL_COLUMN_COUNT:
        sections["level_columns"] = list(LEVEL_COLUMNS_FALLBACK)
        filled.append("level_columns")

    return DocumentSections(**sections), filled  # type: ignore[typeddict-item]


def build_sections(
    *,
    instrument: str,
    timeframe: str,
    scope: str,
    language: str,
    analytics_result: Mapping[str, Any] | None,
    chart: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> tuple[DocumentSections, list[str]]:
    """The model's whole contribution: prose and labels, in `language`.

    Structured output rather than free text — a missing heading label would otherwise
    surface as an empty Word heading rather than as an error.
    """
    result = analytics_result or {}
    prompt = _SECTIONS_PROMPT.format(
        instrument=instrument,
        timeframe=timeframe,
        scope=scope,
        language=language,
        news_summary=result.get("summary") or "- none retrieved",
        sources=_render_sources(result.get("items")),
        chart_status=_chart_status(chart),
        levels=_render_levels((chart or {}).get("levels")),
        commentary=(chart or {}).get("commentary") or "- none",
        column_count=LEVEL_COLUMN_COUNT,
    )
    model = chat_model(AGENT, config).with_structured_output(DocumentSections)
    sections = model.invoke(prompt)
    if not isinstance(sections, dict):
        raise ValueError(f"model returned {type(sections).__name__}, expected the sections dict")
    return complete_sections(sections, instrument=instrument, timeframe=timeframe)


def _runtime_config() -> Mapping[str, Any] | None:
    """The invocation's `configurable`, for the per-agent model override of spec §6.

    Read via `langgraph.config.get_config()` rather than a second node parameter, so the
    entry point keeps the `(state) -> dict` shape §2 requires. Outside a graph run it
    raises, and `None` means "use the defaults".
    """
    try:
        from langgraph.config import get_config

        return get_config()
    except Exception:
        return None


def run(state: AgentState) -> dict[str, Any]:
    """Write the document. Returns a partial update; never raises."""
    started = perf_counter()

    def log(status: str) -> list[dict[str, Any]]:
        return [
            {
                "agent": AGENT,
                "status": status,
                "duration_s": round(perf_counter() - started, 3),
            }
        ]

    try:
        scope = state.get("scope")
        if scope not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {VALID_SCOPES}, got {scope!r}")

        instrument = (state.get("instrument") or "").strip().upper()
        if not instrument:
            raise ValueError("no instrument resolved")
        timeframe = state.get("timeframe") or DEFAULT_TIMEFRAME
        language = state.get("language") or "en"
        analytics_result = state.get("analytics_result")
        chart = state.get("chart")

        sections, filled = build_sections(
            instrument=instrument,
            timeframe=timeframe,
            scope=scope,
            language=language,
            analytics_result=analytics_result,
            chart=chart,
            config=_runtime_config(),
        )

        path = report_path(instrument, now=datetime.now(timezone.utc))
        build_document(
            sections,
            scope=scope,
            path=path,
            chart=chart,
            items=(analytics_result or {}).get("items") or (),
        )

        update: dict[str, Any] = {
            "document": {
                "path": path,
                "format": "docx",
                "title": sections["title"],
                "scope": scope,
                "language": language,
            },
            # A one-line trace, not the prose: `messages` is checkpointed on every
            # superstep, and the document itself is on disk. See spec §4.
            "messages": [AIMessage(content=f"Document written: {sections['title']} -> {path}")],
            "agent_log": log("ok" if not filled else "partial"),
        }
        if filled:
            # The document shipped, but a model that keeps omitting fields is worth knowing
            # about rather than silently papering over.
            update["errors"] = [
                f"{AGENT}: model omitted {', '.join(filled)}; filled from fallbacks"
            ]
        return update

    except Exception as exc:  # noqa: BLE001 - the node contract is "never raise"
        return {
            "document": None,
            "errors": [f"{AGENT}: {exc!r}"],
            "agent_log": log("error"),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the writer agent on its own.")
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--scope", default="news", choices=list(VALID_SCOPES))
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--chart",
        help="path to an existing PNG, for --scope chart/both without running the trader",
    )
    args = parser.parse_args()

    minimal = AgentState(
        instrument=args.instrument,
        timeframe=args.timeframe,
        scope=args.scope,
        language=args.language,
        analytics_result={"summary": "", "items": []},
        chart={"path": args.chart, "levels": [], "commentary": ""} if args.chart else None,
    )
    print(json.dumps(run(minimal), indent=2, default=str))
