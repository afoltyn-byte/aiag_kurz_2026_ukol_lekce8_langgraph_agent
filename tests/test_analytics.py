"""analytics node — spec §4. Tavily and the LLM are stubbed; nothing hits a network.

The load-bearing case is the failure path: `analytics_result` must come back *present
but empty* rather than `None`, because the router reads presence as "produced" and a
`None` would loop the agent until the step limit.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agents import analytics
from src.agents.analytics import NewsItem

ITEMS: list[NewsItem] = [
    NewsItem(
        title="ECB holds rates",
        url="https://example.com/ecb",
        published="2026-08-17",
        snippet="The ECB left its policy rate unchanged.",
    ),
    NewsItem(
        title="Dollar softens",
        url="https://example.com/usd",
        published="2026-08-18",
        snippet="The dollar eased against major peers.",
    ),
]


@pytest.fixture
def stub_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "search_news", lambda *a, **k: list(ITEMS))


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analytics,
        "summarise",
        lambda *a, **k: "ECB held rates (https://example.com/ecb).",
    )


# --- query building -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("EURUSD", "EURUSD"),
        ("EURUSD.pro", "EURUSD"),
        ("XAUUSD_m", "XAUUSD"),
        ("US30-cash", "US30"),
        ("  eurusd.ecn  ", "EURUSD"),
    ],
    ids=["plain", "dot-suffix", "underscore", "dash", "whitespace-and-case"],
)
def test_broker_suffixes_are_stripped(raw: str, expected: str) -> None:
    """Live MT5 symbols carry account-type suffixes that ruin a news search."""
    assert analytics.normalise_symbol(raw) == expected


def test_a_six_letter_symbol_is_split_into_a_currency_pair() -> None:
    query = analytics.build_query("EURUSD.pro")
    assert "EUR/USD" in query
    assert "EURUSD" in query


def test_a_non_pair_symbol_is_used_as_its_own_name() -> None:
    query = analytics.build_query("US30")
    assert query.startswith("US30")
    assert "/" not in query


def test_an_unusable_instrument_is_rejected() -> None:
    with pytest.raises(ValueError, match="no usable symbol"):
        analytics.build_query(".pro")


def test_query_building_is_deterministic() -> None:
    assert analytics.build_query("EURUSD") == analytics.build_query("EURUSD")


# --- tool discovery -----------------------------------------------------------


def test_search_tools_win_by_default() -> None:
    """Research mode is available, not default: a deep-research call would blow the
    30-second timeout."""
    assert analytics.select_search_tool(["tavily_research", "tavily_search"]) == (
        "tavily_search"
    )


def test_research_mode_flips_the_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "TAVILY_RESEARCH", True)
    assert analytics.select_search_tool(["tavily_research", "tavily_search"]) == (
        "tavily_research"
    )


def test_a_server_offering_only_research_still_resolves() -> None:
    assert analytics.select_search_tool(["tavily_research"]) == "tavily_research"


def test_pinned_tool_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "TAVILY_TOOL", "custom_lookup")
    assert analytics.select_search_tool(["tavily_search", "custom_lookup"]) == "custom_lookup"


# --- argument mapping ---------------------------------------------------------


def test_depth_and_topic_are_sent_when_the_tool_exposes_them() -> None:
    schema = {
        "properties": {"query": {}, "max_results": {}, "search_depth": {}, "topic": {}},
        "required": ["query"],
    }
    args = analytics.build_tool_arguments(schema, query="EUR/USD news", max_results=8)
    assert args == {
        "query": "EUR/USD news",
        "max_results": 8,
        "search_depth": analytics.TAVILY_SEARCH_DEPTH,
        "topic": analytics.TAVILY_TOPIC,
    }


def test_a_minimal_tool_gets_only_what_it_declares() -> None:
    schema = {"properties": {"q": {}}, "required": ["q"]}
    assert analytics.build_tool_arguments(schema, query="gold news", max_results=8) == {
        "q": "gold news"
    }


def test_an_unfillable_required_parameter_is_an_error() -> None:
    schema = {"properties": {"query": {}, "api_key": {}}, "required": ["query", "api_key"]}
    with pytest.raises(ValueError, match=r"required parameters \['api_key'\]"):
        analytics.build_tool_arguments(schema, query="x", max_results=8)


# --- payload parsing ----------------------------------------------------------


def test_parse_items_normalises_tavily_field_names() -> None:
    payload = {
        "results": [
            {
                "title": "ECB holds",
                "url": "https://example.com/a",
                "published_date": "2026-08-17",
                "content": "text",
            }
        ]
    }
    assert analytics.parse_items(payload) == [
        NewsItem(
            title="ECB holds",
            url="https://example.com/a",
            published="2026-08-17",
            snippet="text",
        )
    ]


def test_parse_items_accepts_a_bare_list() -> None:
    payload = [{"link": "https://example.com/b", "headline": "H", "text": "T"}]
    assert analytics.parse_items(payload)[0]["url"] == "https://example.com/b"


def test_parse_items_drops_results_without_a_url() -> None:
    """An uncitable item is worse than no item: every claim needs a source."""
    payload = [{"title": "no link"}, {"title": "ok", "url": "https://example.com/c"}]
    assert [i["url"] for i in analytics.parse_items(payload)] == ["https://example.com/c"]


def test_parse_items_deduplicates_by_url() -> None:
    payload = [
        {"url": "https://example.com/d", "title": "first"},
        {"url": "https://example.com/d", "title": "again"},
    ]
    assert len(analytics.parse_items(payload)) == 1


def test_parse_items_respects_the_cap() -> None:
    payload = [{"url": f"https://example.com/{i}"} for i in range(50)]
    assert len(analytics.parse_items(payload, max_items=3)) == 3


def test_a_missing_title_falls_back_to_the_url() -> None:
    assert analytics.parse_items([{"url": "https://example.com/e"}])[0]["title"] == (
        "https://example.com/e"
    )


def test_parse_items_rejects_a_payload_with_no_result_list() -> None:
    with pytest.raises(ValueError, match="no result list"):
        analytics.parse_items({"error": "quota exceeded"})


def test_an_empty_result_list_parses_to_nothing() -> None:
    """Distinct from a malformed payload: the search ran, it just found nothing."""
    assert analytics.parse_items({"results": []}) == []


# --- sourcing guard -----------------------------------------------------------


def test_a_summary_citing_a_given_url_is_sourced() -> None:
    assert analytics.summary_is_sourced("Rates held (https://example.com/ecb).", ITEMS)


def test_a_summary_citing_nothing_is_not_sourced() -> None:
    """Catches a summary written from the model's own knowledge rather than the items."""
    assert not analytics.summary_is_sourced("The ECB held rates this week.", ITEMS)


# --- summary prompt -----------------------------------------------------------


def test_the_prompt_pins_the_model_to_the_retrieved_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeModel:
        def invoke(self, prompt: str) -> Any:
            captured["prompt"] = prompt
            return type("R", (), {"content": "  souhrn  "})()

    monkeypatch.setattr(analytics, "chat_model", lambda *a, **k: FakeModel())

    result = analytics.summarise("EURUSD", "cs", ITEMS)

    assert result == "souhrn"  # stripped
    assert "cs" in captured["prompt"]
    assert "https://example.com/ecb" in captured["prompt"]
    assert "ONLY source of facts" in captured["prompt"]
    assert "do not give trading advice" in captured["prompt"].lower()


def test_no_items_means_no_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_: Any, **__: Any) -> Any:
        raise AssertionError("the model must not be called with nothing to summarise")

    monkeypatch.setattr(analytics, "chat_model", explode)
    assert analytics.summarise("EURUSD", "cs", []) == ""


# --- the node -----------------------------------------------------------------


def test_run_produces_the_full_result_contract(stub_search: None, stub_llm: None) -> None:
    update = analytics.run({"instrument": "eurusd", "language": "cs"})

    result = update["analytics_result"]
    assert set(result) == {"summary", "items"}
    assert result["items"] == ITEMS
    assert result["summary"].startswith("ECB held rates")
    assert "errors" not in update


def test_run_logs_one_ok_entry(stub_search: None, stub_llm: None) -> None:
    entry = analytics.run({"instrument": "EURUSD"})["agent_log"][0]
    assert entry["agent"] == "analytics"
    assert entry["status"] == "ok"
    assert isinstance(entry["duration_s"], float)


def test_run_returns_only_keys_this_agent_owns(stub_search: None, stub_llm: None) -> None:
    update = analytics.run({"instrument": "EURUSD"})
    assert set(update) <= {"analytics_result", "errors", "agent_log"}


def test_run_does_not_mutate_the_state(stub_search: None, stub_llm: None) -> None:
    state: Any = {"instrument": "EURUSD", "language": "cs"}
    snapshot = dict(state)
    analytics.run(state)
    assert state == snapshot


def test_language_defaults_to_english(stub_search: None, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        analytics,
        "summarise",
        lambda instrument, language, items: captured.setdefault("lang", language) or "s",
    )
    analytics.run({"instrument": "EURUSD"})
    assert captured["lang"] == "en"


def test_a_failed_summary_keeps_the_items(
    stub_search: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer can still list sources; throwing the fetch away would be wasteful."""

    def boom(*_: Any, **__: Any) -> str:
        raise RuntimeError("no API key")

    monkeypatch.setattr(analytics, "summarise", boom)

    update = analytics.run({"instrument": "EURUSD"})

    assert update["analytics_result"]["items"] == ITEMS
    assert update["analytics_result"]["summary"] == ""
    assert any("summary failed" in e for e in update["errors"])
    assert update["agent_log"][0]["status"] == "partial"


def test_an_unsourced_summary_is_flagged_but_kept(
    stub_search: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analytics, "summarise", lambda *a, **k: "Rates were held.")

    update = analytics.run({"instrument": "EURUSD"})

    assert update["analytics_result"]["summary"] == "Rates were held."
    assert any("cites none of the retrieved URLs" in e for e in update["errors"])


def test_a_dead_tavily_yields_an_empty_but_present_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule that keeps the graph terminating: present-but-empty, never None."""

    def boom(*_: Any, **__: Any) -> list[NewsItem]:
        raise TimeoutError("Tavily did not answer")

    monkeypatch.setattr(analytics, "search_news", boom)

    update = analytics.run({"instrument": "EURUSD"})

    assert update["analytics_result"] == {"summary": "", "items": []}
    assert update["analytics_result"] is not None
    assert any("Tavily did not answer" in e for e in update["errors"])
    assert update["agent_log"][0]["status"] == "error"


def test_a_search_that_finds_nothing_is_reported_but_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analytics, "search_news", lambda *a, **k: [])

    update = analytics.run({"instrument": "EURUSD"})

    assert update["analytics_result"] == {"summary": "", "items": []}
    assert any("no usable items" in e for e in update["errors"])
    assert update["agent_log"][0]["status"] == "empty"


@pytest.mark.parametrize("state", [{}, {"instrument": "  "}], ids=["absent", "blank"])
def test_a_missing_instrument_is_reported_without_raising(state: Any) -> None:
    update = analytics.run(state)
    assert update["analytics_result"] == {"summary": "", "items": []}
    assert any("no instrument" in e for e in update["errors"])


def test_search_without_a_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TAVILY_API_KEY is not set"):
        analytics.search_news("EURUSD")


def test_the_empty_result_constant_is_never_handed_out_by_reference() -> None:
    """A shared mutable default in state would be a cross-run aliasing bug."""
    update = analytics.run({})
    assert update["analytics_result"] is not analytics.EMPTY_RESULT


def test_cli_json_dump_is_serialisable(stub_search: None, stub_llm: None) -> None:
    json.dumps(analytics.run({"instrument": "EURUSD"}), default=str)


# --- the server's own enum ----------------------------------------------------


def test_topic_is_dropped_when_the_server_forbids_it() -> None:
    """The exact failure from the first real run against the hosted Tavily MCP: it declares
    `topic` as Literal['general'], and `news` came back as a pydantic validation error."""
    schema = {
        "properties": {"query": {}, "topic": {"enum": ["general"]}},
        "required": ["query"],
    }

    arguments = analytics.build_tool_arguments(schema, query="EUR/USD news", max_results=8)

    assert arguments["query"] == "EUR/USD news"
    assert arguments.get("topic") in (None, "general")


def test_the_configured_topic_is_one_the_hosted_server_accepts() -> None:
    """Sending the value the server allows beats sending nothing at all."""
    assert analytics.TAVILY_TOPIC == "general"


def test_an_unsupported_search_depth_is_dropped_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "TAVILY_SEARCH_DEPTH", "extreme")
    schema = {
        "properties": {"query": {}, "search_depth": {"enum": ["basic", "advanced"]}},
        "required": ["query"],
    }

    assert analytics.build_tool_arguments(schema, query="x", max_results=8) == {"query": "x"}
