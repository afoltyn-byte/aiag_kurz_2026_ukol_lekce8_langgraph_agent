"""analytics agent — spec §4 `analytics`.

Searches for news about the instrument over the Tavily MCP server and summarises what
came back. Reads `instrument` and `language`; writes `analytics_result`, `errors`,
`agent_log`.

Read-only: nothing is written outside state, which is what makes a retry free.

The model summarises **retrieved items and nothing else** — no synthesis beyond what
was fetched, and every claim carries a source URL from `items`. The search itself, the
query, and the tool choice are all deterministic; the model never decides what to look
for.

On failure `analytics_result` is `{"summary": "", "items": []}` — present but empty, not
`None`. That distinction is load-bearing: the router treats a present artifact as
produced, so a dead Tavily ends in a document that notes the gap rather than an endless
retry loop (spec §5 rule 4).
"""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence, TypedDict

from src.agent.config import (
    TAVILY_MAX_ITEMS,
    TAVILY_RESEARCH,
    TAVILY_RESEARCH_TOOL_CANDIDATES,
    TAVILY_SEARCH_DEPTH,
    TAVILY_SEARCH_TOOL_CANDIDATES,
    TAVILY_TIMEOUT_S,
    TAVILY_TOOL,
    TAVILY_TOPIC,
    chat_model,
    tavily_api_key,
    tavily_mcp_url,
)
from src.agent.state import AgentState
from src.agents.mcp_client import call_mcp_tool, match_parameters, select_tool

AGENT = "analytics"

# Parameter-name aliases on the discovered tool, most explicit first.
_QUERY_KEYS = ("query", "q", "search_query", "question", "input")
_MAX_RESULTS_KEYS = ("max_results", "maxresults", "limit", "count", "num_results", "n")
_DEPTH_KEYS = ("search_depth", "depth", "mode")
_TOPIC_KEYS = ("topic", "category")

# Where a JSON payload might keep the result list.
_RESULT_KEYS = ("results", "items", "articles", "data", "hits", "documents", "sources")

# Per-field spellings inside one result.
_ITEM_FIELDS: dict[str, tuple[str, ...]] = {
    "title": ("title", "name", "headline"),
    "url": ("url", "link", "source_url", "href"),
    "published": ("published", "published_date", "published_time", "date", "publishedat"),
    "snippet": ("snippet", "content", "text", "summary", "description", "raw_content"),
}

_FX_PAIR = re.compile(r"^[A-Z]{6}$")


class NewsItem(TypedDict):
    title: str
    url: str
    published: str
    snippet: str


class AnalyticsResult(TypedDict):
    summary: str
    items: list[NewsItem]


EMPTY_RESULT: AnalyticsResult = {"summary": "", "items": []}


# --------------------------------------------------------------------------
# query: deterministic, no model
# --------------------------------------------------------------------------


def normalise_symbol(instrument: str) -> str:
    """Strip the broker suffix from an MT5 symbol.

    Live symbols carry account-type suffixes — `EURUSD.pro`, `XAUUSD_m`, `US30-cash` —
    which are meaningless to a news search and actively harm it.
    """
    return re.split(r"[^A-Za-z0-9]", instrument.strip().upper(), maxsplit=1)[0]


def build_query(instrument: str) -> str:
    """A news query from a broker symbol. Pure string work — the model is not consulted.

    A six-letter symbol is a currency pair, so it is split (`EURUSD` -> `EUR/USD`);
    anything else is passed through as its own name.
    """
    symbol = normalise_symbol(instrument)
    if not symbol:
        raise ValueError(f"instrument {instrument!r} has no usable symbol")
    if _FX_PAIR.match(symbol):
        return f"{symbol[:3]}/{symbol[3:]} {symbol} forex market news"
    return f"{symbol} market news"


# --------------------------------------------------------------------------
# Tavily MCP
# --------------------------------------------------------------------------


def tool_candidates() -> tuple[str, ...]:
    """Research tools first when research mode is on, search tools otherwise.

    Both lists are kept so a server offering only one still resolves.
    """
    if TAVILY_RESEARCH:
        return tuple(TAVILY_RESEARCH_TOOL_CANDIDATES) + tuple(TAVILY_SEARCH_TOOL_CANDIDATES)
    return tuple(TAVILY_SEARCH_TOOL_CANDIDATES) + tuple(TAVILY_RESEARCH_TOOL_CANDIDATES)


def select_search_tool(tool_names: Iterable[str]) -> str:
    """The search tool. `TAVILY_TOOL` pins an exact name; otherwise discovery."""
    return select_tool(
        tool_names, tool_candidates(), pinned=TAVILY_TOOL, label="Tavily search tool"
    )


def build_tool_arguments(
    input_schema: Mapping[str, Any],
    *,
    query: str,
    max_results: int,
) -> dict[str, Any]:
    """Map the query onto whatever the tool calls its parameters.

    Depth and topic are offered but optional: a server that does not expose them simply
    does not get them, whereas a required parameter we cannot fill is an error.
    """
    return match_parameters(
        input_schema,
        [
            (_QUERY_KEYS, query),
            (_MAX_RESULTS_KEYS, max_results),
            (_DEPTH_KEYS, TAVILY_SEARCH_DEPTH),
            (_TOPIC_KEYS, TAVILY_TOPIC),
        ],
        label="Tavily tool",
    )


def _pick(row: Mapping[str, Any], field: str, default: str = "") -> str:
    lowered = {k.lower(): v for k, v in row.items()}
    for alias in _ITEM_FIELDS[field]:
        value = lowered.get(alias)
        if value not in (None, ""):
            return str(value).strip()
    return default


def parse_items(payload: Any, *, max_items: int = TAVILY_MAX_ITEMS) -> list[NewsItem]:
    """Normalise the server's response into `items`, deduplicated by URL.

    A result with no URL is dropped rather than kept: the summary has to cite a source
    for every claim, so an uncitable item is worse than no item.
    """
    rows = payload
    if isinstance(rows, Mapping):
        for key in _RESULT_KEYS:
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
        else:
            raise ValueError(f"no result list in payload keys {sorted(rows)}")

    if not isinstance(rows, list):
        raise ValueError(f"unexpected Tavily payload type {type(rows).__name__}")

    items: list[NewsItem] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        url = _pick(row, "url")
        if not url or url in seen:
            continue
        seen.add(url)
        items.append(
            NewsItem(
                title=_pick(row, "title", default=url),
                url=url,
                published=_pick(row, "published"),
                snippet=_pick(row, "snippet"),
            )
        )
        if len(items) >= max_items:
            break
    return items


def search_news(instrument: str, *, max_items: int = TAVILY_MAX_ITEMS) -> list[NewsItem]:
    """Search Tavily over MCP and normalise the results. Blocking; raises on failure."""
    key = tavily_api_key()
    if not key:
        raise ValueError("TAVILY_API_KEY is not set in the environment")

    query = build_query(instrument)
    tool_name, payload = call_mcp_tool(
        tavily_mcp_url(),
        tool_candidates(),
        lambda schema: build_tool_arguments(schema, query=query, max_results=max_items),
        timeout_s=TAVILY_TIMEOUT_S,
        pinned=TAVILY_TOOL,
        # Header rather than a query parameter: a key in a URL ends up in logs and
        # proxy history. Set TAVILY_MCP_URL explicitly if a deployment needs it inline.
        headers={"Authorization": f"Bearer {key}"},
        label="Tavily search tool",
    )
    try:
        return parse_items(payload, max_items=max_items)
    except ValueError as exc:
        raise ValueError(f"{exc} (tool {tool_name!r})") from exc


# --------------------------------------------------------------------------
# summary: the model's only job, and it stays inside the retrieved set
# --------------------------------------------------------------------------

_SUMMARY_PROMPT = """You are summarising retrieved news for a broker-side analyst.

Instrument: {instrument}
Write in this language (ISO 639-1 code): {language}

Retrieved items — these are your ONLY source of facts:
{items}

Write a summary of what these items report about the instrument. Rules:
- Use only what the items above state. Add no background, context, figures or events
  that are not in them. If they say little, write little.
- Every claim must be followed by the URL of the item it came from, in parentheses.
- Do not give trading advice, recommendations or price targets.
- Write in the language identified by the code above."""


def summarise(instrument: str, language: str, items: Sequence[NewsItem]) -> str:
    """Summarise the retrieved items, in `language`, with a source URL per claim.

    The items are the model's entire factual universe here — it is summarising a fetched
    set, not answering from its own knowledge.
    """
    if not items:
        return ""

    rendered = "\n\n".join(
        f"[{i}] {item['title']}\n    url: {item['url']}\n"
        f"    published: {item['published'] or 'unknown'}\n"
        f"    excerpt: {item['snippet']}"
        for i, item in enumerate(items, start=1)
    )
    prompt = _SUMMARY_PROMPT.format(
        instrument=instrument, language=language, items=rendered
    )
    response = chat_model(AGENT, _runtime_config()).invoke(prompt)
    return str(response.content).strip()


def summary_is_sourced(summary: str, items: Sequence[NewsItem]) -> bool:
    """Does the summary cite at least one of the URLs it was given?

    A cheap, deterministic check on the "every claim carries a source URL" rule. It
    cannot verify per-claim attribution, but it does catch a summary written entirely
    from the model's own knowledge — which is the failure that matters.
    """
    return any(item["url"] in summary for item in items)


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


# --------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------


def run(state: AgentState) -> dict[str, Any]:
    """Search, then summarise. Returns a partial update; never raises."""
    started = perf_counter()
    errors: list[str] = []

    def log(status: str) -> list[dict[str, Any]]:
        return [
            {
                "agent": AGENT,
                "status": status,
                "duration_s": round(perf_counter() - started, 3),
            }
        ]

    try:
        instrument = (state.get("instrument") or "").strip().upper()
        if not instrument:
            raise ValueError("no instrument resolved")
        language = state.get("language") or "en"

        items = search_news(instrument)
        if not items:
            # A successful search that found nothing is not an error, but the writer has
            # to be able to tell that apart from a search that never ran.
            return {
                "analytics_result": AnalyticsResult(summary="", items=[]),
                "errors": [f"{AGENT}: search returned no usable items for {instrument}"],
                "agent_log": log("empty"),
            }

        # The summary is the optional half: retrieved items are worth keeping even if
        # the model call fails, because the writer can still list the sources.
        summary = ""
        try:
            summary = summarise(instrument, language, items)
            if summary and not summary_is_sourced(summary, items):
                errors.append(f"{AGENT}: summary cites none of the retrieved URLs")
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(f"{AGENT}: summary failed, items kept: {exc!r}")

        update: dict[str, Any] = {
            "analytics_result": AnalyticsResult(summary=summary, items=items),
            "agent_log": log("ok" if not errors else "partial"),
        }
        if errors:
            update["errors"] = errors
        return update

    except Exception as exc:  # noqa: BLE001 - the node contract is "never raise"
        return {
            "analytics_result": dict(EMPTY_RESULT),
            "errors": errors + [f"{AGENT}: {exc!r}"],
            "agent_log": log("error"),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the analytics agent on its own.")
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="fetch and list items only, no LLM call and no OPENAI_API_KEY needed",
    )
    args = parser.parse_args()

    if args.no_summary:
        # Rebind the module attribute; run() resolves the name as a global at call time.
        globals()["summarise"] = lambda *_, **__: ""

    minimal = AgentState(instrument=args.instrument, language=args.language)
    print(json.dumps(run(minimal), indent=2, default=str))
