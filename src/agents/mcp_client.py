"""Shared MCP plumbing for the `trader` and `analytics` agents.

Neither the MT5 nor the Tavily server has its tool contract defined in the spec, so
both agents discover the tool they need at runtime. The *transport dance* (connect,
initialise, list, match arguments, call, unwrap the payload) is identical for both and
lives here; the alias tables and the payload shapes stay in the agents, because those
genuinely differ.

Everything here is deterministic string matching — **no model is involved in choosing
a tool or filling an argument.** A wrong tool name sent to a trading server is worse
than a loud failure, so an unfillable *required* parameter raises rather than guesses.

This module raises. The agents catch and turn failures into `errors` entries.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Iterable, Mapping, Sequence


def select_tool(
    tool_names: Iterable[str],
    candidates: Sequence[str],
    *,
    pinned: str | None = None,
    label: str = "tool",
) -> str:
    """The first tool whose name contains a candidate fragment, candidates in order.

    Ties break on the sorted tool name, so the same server always yields the same
    choice — discovery has to be reproducible or a checkpoint replay could pick a
    different tool than the original run.
    """
    names = sorted(tool_names)
    if pinned:
        if pinned not in names:
            raise ValueError(f"pinned {label} {pinned!r} not offered by the server: {names}")
        return pinned

    for fragment in candidates:
        for name in names:
            if fragment in name.lower():
                return name
    raise ValueError(f"no {label} matching {list(candidates)} among {names}")


def match_parameters(
    input_schema: Mapping[str, Any],
    wanted: Sequence[tuple[Sequence[str], Any]],
    *,
    label: str = "tool",
) -> dict[str, Any]:
    """Map values onto whatever the tool actually calls its parameters.

    `wanted` is an ordered list of `(aliases, value)`. Aliases are matched
    case-insensitively against the schema's property names, most explicit first.
    Parameters the schema declares but we have no value for are left alone; a *required*
    one we cannot fill is an error.

    A `value` may instead be a **callable** taking the matched property's own schema, for
    parameters whose encoding depends on how the server declared them — a timestamp wanted
    as an ISO string by one server and as epoch seconds by another.

    The schema's **`enum` is honoured**, which is not cosmetic: sending a value the server
    declared invalid produces a validation error from inside the server, where it reads as a
    mysterious failure rather than as our mistake. An optional parameter whose value is out
    of range is dropped; a required one raises here, naming what was allowed.
    """
    properties: Mapping[str, Any] = input_schema.get("properties") or {}
    required: Sequence[str] = input_schema.get("required") or []

    arguments: dict[str, Any] = {}
    for aliases, value in wanted:
        if value is None:
            continue

        match = None
        for alias in aliases:
            match = next((p for p in properties if p.lower() == alias), None)
            if match is not None:
                break
        if match is None:
            continue

        property_schema = properties.get(match) or {}
        resolved = value(property_schema) if callable(value) else value
        if resolved is None:
            continue

        allowed = property_schema.get("enum")
        if allowed and resolved not in allowed:
            if match in required:
                raise ValueError(
                    f"{label} requires {match!r} to be one of {allowed}, "
                    f"and {resolved!r} is not"
                )
            continue  # optional and unsupported: the server declared it, we respect it

        arguments[match] = resolved

    unmatched = [p for p in required if p not in arguments]
    if unmatched:
        raise ValueError(f"cannot fill required parameters {unmatched} of the {label}")
    return arguments


def extract_payload(result: Any, tool_name: str) -> Any:
    """`structured_content` when the server provides it, else the text blocks as JSON."""
    if getattr(result, "is_error", False):
        raise RuntimeError(f"tool {tool_name!r} returned an error: {result.content!r}")

    structured = getattr(result, "structured_content", None)
    if structured:
        return structured

    text = "".join(
        block.text for block in getattr(result, "content", []) if hasattr(block, "text")
    )
    if not text.strip():
        raise RuntimeError(f"tool {tool_name!r} returned no usable content")
    return json.loads(text)


async def _call_async(
    url: str,
    candidates: Sequence[str],
    pinned: str | None,
    build_arguments: Callable[[Mapping[str, Any]], dict[str, Any]],
    timeout_s: float,
    headers: Mapping[str, str] | None,
    label: str,
) -> tuple[str, Any]:
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    # A client is only built when headers are needed; otherwise the transport makes its
    # own. `timeout` bounds the connect/read at the HTTP layer, `read_timeout_seconds`
    # bounds the MCP request — a hung server has to hit one of them.
    http_client = (
        httpx2.AsyncClient(headers=dict(headers), timeout=timeout_s) if headers else None
    )
    try:
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timeout_s) as session:
                await session.initialize()

                listing = await session.list_tools()
                tools = {tool.name: tool for tool in listing.tools}
                name = select_tool(tools, candidates, pinned=pinned, label=label)
                arguments = build_arguments(tools[name].input_schema or {})

                result = await session.call_tool(
                    name, arguments, read_timeout_seconds=timeout_s
                )
        return name, extract_payload(result, name)
    finally:
        if http_client is not None:
            await http_client.aclose()


def call_mcp_tool(
    url: str,
    candidates: Sequence[str],
    build_arguments: Callable[[Mapping[str, Any]], dict[str, Any]],
    *,
    timeout_s: float,
    pinned: str | None = None,
    headers: Mapping[str, str] | None = None,
    label: str = "tool",
) -> tuple[str, Any]:
    """Blocking one-shot MCP call. Returns `(tool_name, payload)`.

    The tool name comes back so a failure downstream can say *which* tool was chosen —
    the single most useful piece of information when discovery guesses wrong.

    `asyncio.run` is correct here: LangGraph runs a sync node in a worker thread, so
    there is no running loop to clash with.
    """
    return asyncio.run(
        _call_async(url, candidates, pinned, build_arguments, timeout_s, headers, label)
    )
