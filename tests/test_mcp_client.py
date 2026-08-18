"""Shared MCP plumbing. The transport itself is not exercised here — no server.

`select_tool` and `match_parameters` are reached through the agents' own wrappers too;
what has no other coverage is `extract_payload`, which is where a server's error
response either becomes a clear exception or gets silently mistaken for data.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.mcp_client import extract_payload, match_parameters, select_tool


class Block:
    def __init__(self, text: str) -> None:
        self.text = text


class Result:
    def __init__(
        self,
        *,
        content: list[Any] | None = None,
        structured_content: Any = None,
        is_error: bool = False,
    ) -> None:
        self.content = content or []
        self.structured_content = structured_content
        self.is_error = is_error


# --- select_tool --------------------------------------------------------------


def test_candidate_order_beats_alphabetical_order() -> None:
    assert select_tool(["b_search", "a_rates"], ("rates", "search")) == "a_rates"


def test_ties_break_on_the_sorted_tool_name() -> None:
    """Discovery has to be reproducible: a checkpoint replay must pick the same tool."""
    names = ["z_rates", "a_rates"]
    assert select_tool(names, ("rates",)) == "a_rates"
    assert select_tool(reversed(names), ("rates",)) == "a_rates"


def test_matching_is_case_insensitive() -> None:
    assert select_tool(["MT5_CopyRates"], ("copyrates",)) == "MT5_CopyRates"


def test_no_match_names_both_sides_of_the_mismatch() -> None:
    with pytest.raises(ValueError) as excinfo:
        select_tool(["order_send"], ("rates",), label="MT5 OHLC tool")
    message = str(excinfo.value)
    assert "MT5 OHLC tool" in message
    assert "rates" in message
    assert "order_send" in message  # what the server actually offered


def test_a_pinned_name_skips_discovery_entirely() -> None:
    assert select_tool(["copy_rates", "weird_name"], ("rates",), pinned="weird_name") == (
        "weird_name"
    )


def test_a_pinned_name_that_is_absent_fails_loudly() -> None:
    with pytest.raises(ValueError, match="not offered by the server"):
        select_tool(["copy_rates"], ("rates",), pinned="nope")


# --- match_parameters --------------------------------------------------------


def test_earlier_aliases_win() -> None:
    schema = {"properties": {"q": {}, "query": {}}}
    assert match_parameters(schema, [(("query", "q"), "x")]) == {"query": "x"}


def test_the_servers_own_capitalisation_is_preserved() -> None:
    schema = {"properties": {"Symbol": {}}}
    assert match_parameters(schema, [(("symbol",), "EURUSD")]) == {"Symbol": "EURUSD"}


def test_a_none_value_is_never_sent() -> None:
    schema = {"properties": {"topic": {}}}
    assert match_parameters(schema, [(("topic",), None)]) == {}


def test_properties_we_have_no_value_for_are_left_alone() -> None:
    schema = {"properties": {"query": {}, "include_images": {}}, "required": ["query"]}
    assert match_parameters(schema, [(("query",), "x")]) == {"query": "x"}


def test_an_unfillable_required_parameter_raises_rather_than_guessing() -> None:
    schema = {"properties": {"query": {}, "account": {}}, "required": ["query", "account"]}
    with pytest.raises(ValueError, match=r"required parameters \['account'\]"):
        match_parameters(schema, [(("query",), "x")], label="Tavily tool")


def test_an_empty_schema_is_a_legitimate_no_argument_tool() -> None:
    assert match_parameters({}, [(("query",), "x")]) == {}


# --- extract_payload ---------------------------------------------------------


def test_structured_content_is_preferred() -> None:
    result = Result(
        structured_content={"bars": [1]}, content=[Block('{"bars": [999]}')]
    )
    assert extract_payload(result, "t") == {"bars": [1]}


def test_text_blocks_are_concatenated_then_parsed() -> None:
    result = Result(content=[Block('{"a":'), Block(" 1}")])
    assert extract_payload(result, "t") == {"a": 1}


def test_an_error_flag_becomes_an_exception_naming_the_tool() -> None:
    """Otherwise an error object parses as 'data' and surfaces as a confusing bug later."""
    result = Result(content=[Block("rate limited")], is_error=True)
    with pytest.raises(RuntimeError, match="copy_rates.*error"):
        extract_payload(result, "copy_rates")


def test_empty_content_is_an_exception_not_an_empty_result() -> None:
    with pytest.raises(RuntimeError, match="no usable content"):
        extract_payload(Result(content=[Block("   ")]), "t")


def test_non_json_text_surfaces_as_a_decode_error() -> None:
    import json

    with pytest.raises(json.JSONDecodeError):
        extract_payload(Result(content=[Block("<html>nope</html>")]), "t")


def test_blocks_without_text_are_skipped() -> None:
    """Image or resource blocks alongside the JSON must not break parsing."""
    result = Result(content=[object(), Block('{"a": 1}')])
    assert extract_payload(result, "t") == {"a": 1}


# --- the schema's enum --------------------------------------------------------


def test_an_optional_value_the_schema_forbids_is_dropped() -> None:
    """Found on the first real run: the hosted Tavily MCP declares `topic` as a
    single-value enum, and sending `news` produced a pydantic error from inside the server
    where it read as a mystery rather than as our mistake."""
    schema = {"properties": {"query": {}, "topic": {"enum": ["general"]}}, "required": ["query"]}

    arguments = match_parameters(schema, [(("query",), "x"), (("topic",), "news")])

    assert arguments == {"query": "x"}


def test_an_in_range_value_passes_through() -> None:
    schema = {"properties": {"topic": {"enum": ["general", "news"]}}}
    assert match_parameters(schema, [(("topic",), "news")]) == {"topic": "news"}


def test_a_required_value_the_schema_forbids_raises_with_the_allowed_set() -> None:
    """Dropping a required parameter would just fail one line later with a worse message."""
    schema = {"properties": {"mode": {"enum": ["a", "b"]}}, "required": ["mode"]}

    with pytest.raises(ValueError, match=r"'mode' to be one of \['a', 'b'\]"):
        match_parameters(schema, [(("mode",), "c")], label="Some tool")


def test_a_property_without_an_enum_is_not_second_guessed() -> None:
    schema = {"properties": {"query": {"type": "string"}}}
    assert match_parameters(schema, [(("query",), "anything")]) == {"query": "anything"}


# --- callable values ----------------------------------------------------------


def test_a_callable_value_receives_the_matched_property_schema() -> None:
    """For parameters whose encoding depends on how the server typed them."""
    seen: list[Any] = []

    def encode(property_schema: Any) -> str:
        seen.append(property_schema)
        return "encoded"

    schema = {"properties": {"when": {"type": "string", "format": "date-time"}}}

    assert match_parameters(schema, [(("when",), encode)]) == {"when": "encoded"}
    assert seen == [{"type": "string", "format": "date-time"}]


def test_a_callable_is_not_invoked_for_a_parameter_the_tool_lacks() -> None:
    def explode(_: Any) -> str:
        raise AssertionError("must not be called")

    assert match_parameters({"properties": {"other": {}}}, [(("when",), explode)]) == {}


def test_a_callable_returning_none_sends_nothing() -> None:
    schema = {"properties": {"when": {}}}
    assert match_parameters(schema, [(("when",), lambda _: None)]) == {}
