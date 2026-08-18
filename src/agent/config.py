"""Per-agent models, limits and paths — spec §6.

No inline literals for any of these elsewhere in the codebase. Secrets are read from
the environment on demand and never stored here; `.env` holds `op://` references and
the process is launched with `op run --env-file=.env -- ...`.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from langchain_openai import ChatOpenAI

# --- models -------------------------------------------------------------------
# Provider is OpenAI. One constant for the default so it has a single home; agents that are
# happy with it point at it rather than repeating the id.
#
# The supervisor is deliberately NOT on the default, and this is a finding rather than a
# preference. On `gpt-5.4-nano` it read "co je nového na eurodolaru a jak vypadá graf" —
# which names both the instrument and the scope — correctly on some runs and returned nulls
# on others, sending the user to `clarify` for something the sentence already said. The
# router did the right thing with a null scope (§4: guessing is worse than asking), so the
# graph behaved; the model simply could not read reliably enough.
#
# The other three stayed on nano and are fine: summarising retrieved items, describing
# already-computed levels and filling a section schema are mechanical next to deciding what
# a free-text request actually asked for. That is the one job here needing comprehension,
# so it is the one job that pays for a bigger model.

DEFAULT_MODEL = "gpt-5.4-nano"
SUPERVISOR_MODEL = "gpt-5.4-mini"

AGENT_MODELS: dict[str, str] = {
    "supervisor": SUPERVISOR_MODEL,
    "analytics": DEFAULT_MODEL,
    "trader": DEFAULT_MODEL,
    "writer": DEFAULT_MODEL,
}

# NOTE: inert for `gpt-5*` reasoning models. langchain_openai's `validate_temperature`
# pops the parameter for any model id starting with "gpt-5" that is not "-chat",
# unless reasoning_effort="none" — silently: no warning, no exception, and the built
# client just reports `temperature=None`. Kept because it records the intent per
# agent and goes live again under a "-chat" or non-gpt-5 model. See spec §6.
AGENT_TEMPERATURE: dict[str, float] = {
    "supervisor": 0.0,
    "analytics": 0.0,
    "trader": 0.0,
    "writer": 0.3,
}

# --- limits and paths ---------------------------------------------------------

MAX_STEPS = 12
MAX_CLARIFICATIONS = 2
RECURSION_LIMIT = 40  # > MAX_STEPS * 2, so step_limit fires before GraphRecursionError
MT5_TIMEOUT_S = 20
TAVILY_TIMEOUT_S = 30
CHART_DIR = "outputs/charts"
REPORT_DIR = "outputs/reports"
CHART_RETENTION_DAYS = 7
DEFAULT_TIMEFRAME = "H1"

# Whitelist for the supervisor's structured output. A timeframe the platform does not
# know would fail deep inside the MT5 call; rejecting it here costs nothing.
MT5_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")

# --- charting.py: level derivation and rendering -------------------------------
# The zone tolerance is expressed in ATRs, not pips: EURUSD, XAUUSD and BTCUSD differ
# by orders of magnitude, so a fixed pip tolerance is meaningless across them.

PIVOT_LOOKBACK = 3  # bars each side of a fractal pivot
ATR_PERIOD = 14
ZONE_ATR_MULT = 0.5  # cluster spread tolerance, in ATRs
MAX_ZONES_PER_SIDE = 3
ZONE_STRENGTH_SATURATION = 4  # touches at which strength reaches 1.0
CHART_DPI = 130
CHART_FIGSIZE = (12.0, 6.5)

# --- trader.py: MT5 MCP access -------------------------------------------------

MT5_BAR_COUNT = 200  # bars requested per chart

# Some MT5 MCP tools take a bar count, others a datetime range. The range is derived from
# the count and the timeframe, so both shapes are served from one setting.
MT5_TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3_600,
    "H4": 14_400,
    "D1": 86_400,
    "W1": 604_800,
    "MN1": 2_592_000,  # 30 days; only ever used to size a lookback window
}

# Wall-clock span is not bar count: markets close. 200 H1 bars span 8+ calendar days, which
# contains two weekends of nothing, and 200 M1 bars requested on a Sunday would span a
# window with no ticks at all. The range is padded so the frame still arrives long enough
# for ATR(14) and the pivot window — a short frame is a ValueError, not a small chart.
MT5_RANGE_PADDING = 3.0

# Broker server offset from UTC. MT5 servers usually run UTC+2/+3. This does two jobs:
#
# 1. A bar timestamp that arrives as a *naive* string is server time, not UTC — MT5 just
#    prints the server clock — so it is stamped with this offset to become a true epoch.
# 2. `derive_levels` shifts back by the same offset to decide which bars share a session,
#    because an FX day does not break at midnight UTC.
#
# The two cancel for session grouping, so sessions land on server midnight whatever this is
# set to. What it fixes is the absolute timestamps: left at 0 against a UTC+3 server, every
# bar time is three hours off.
MT5_SERVER_UTC_OFFSET_H = 3.0

# The MT5 MCP server's tool contract is not part of the spec, so the OHLC tool is
# discovered at runtime: the first tool whose name contains one of these fragments,
# tried in order. Set MT5_OHLC_TOOL to pin an exact name and skip discovery.
MT5_OHLC_TOOL: str | None = None
MT5_OHLC_TOOL_CANDIDATES = (
    "copy_rates",
    "get_rates",
    "rates",
    "ohlc",
    "candle",
    "bars",
    "history",
)

# Auth for the MT5 MCP server, and it is **optional**: the server the spec describes runs
# on localhost, where a key is usually pointless. With MT5_API_KEY unset no auth header is
# sent at all and a plain local server keeps working.
#
# The header shape is configurable because MCP has no convention for this. `Bearer` is the
# common case; a server wanting a raw key under its own header is served by
# MT5_AUTH_HEADER="X-API-Key" with MT5_AUTH_SCHEME="".
MT5_AUTH_HEADER = "Authorization"
MT5_AUTH_SCHEME = "Bearer"

# --- analytics.py: Tavily MCP access -------------------------------------------

# Tavily's hosted MCP endpoint. Override via the environment when self-hosting, or to
# embed the key as a query parameter for a deployment that wants it that way; by
# default the key travels in an Authorization header instead.
TAVILY_MCP_URL_DEFAULT = "https://mcp.tavily.com/mcp/"

TAVILY_MAX_ITEMS = 8
TAVILY_SEARCH_DEPTH = "advanced"  # Tavily's own term for its deeper search

# "general", not "news": the hosted Tavily MCP declares `topic` as a single-value enum
# (`Literal['general']`) even though the REST API accepts more. `match_parameters` would
# drop an out-of-enum value anyway, but sending the one the server accepts is better than
# sending nothing. The query itself already carries the news framing.
TAVILY_TOPIC = "general"

# Research mode is *available*, not the default: a deep-research call routinely runs for
# minutes and TAVILY_TIMEOUT_S is 30, so making it the default would time out most runs.
# Flip this on for a deployment that raises the timeout too.
TAVILY_RESEARCH = False

TAVILY_TOOL = None  # pin an exact tool name, or None to discover
TAVILY_SEARCH_TOOL_CANDIDATES = ("tavily_search", "search", "web_search", "news")
TAVILY_RESEARCH_TOOL_CANDIDATES = ("tavily_research", "research", "deep", "extract")


def resolve_model(agent: str, config: Mapping[str, Any] | None = None) -> str:
    """`configurable.models[agent]` if the caller overrode it, else the default.

    Lets a single invocation swap one agent's model without a code change:

        config={"configurable": {"thread_id": "...", "models": {"trader": "gpt-5.4"}}}
    """
    configurable = (config or {}).get("configurable") or {}
    overrides = configurable.get("models") or {}
    return overrides.get(agent) or AGENT_MODELS[agent]


def resolve_temperature(agent: str) -> float:
    return AGENT_TEMPERATURE[agent]


def chat_model(agent: str, config: Mapping[str, Any] | None = None) -> ChatOpenAI:
    """The one place an OpenAI client is constructed.

    Reads `OPENAI_API_KEY` from the environment via the SDK's own lookup, so the key
    never passes through this codebase. Construction *can* raise (missing key), which
    is why every caller sits inside the node's try/except: the failure becomes an
    `errors` entry and control returns to the supervisor (spec §4).
    """
    return ChatOpenAI(
        model=resolve_model(agent, config),
        temperature=resolve_temperature(agent),
    )


# --- secrets: environment only, never committed, never defaulted ---------------
# These return None rather than raising: a node must turn a missing value into an
# `errors` entry and hand control back to the supervisor (spec §4, "no agent
# raises"), not kill the run at import time.


def openai_api_key() -> str | None:
    """Only for pre-flight checks — `chat_model()` lets the SDK read it itself."""
    return os.environ.get("OPENAI_API_KEY")


def tavily_api_key() -> str | None:
    return os.environ.get("TAVILY_API_KEY")


def tavily_mcp_url() -> str:
    """Endpoint for the Tavily MCP server; the hosted one unless overridden."""
    return os.environ.get("TAVILY_MCP_URL") or TAVILY_MCP_URL_DEFAULT


def mt5_mcp_url() -> str | None:
    """Streamable-HTTP endpoint of the local MT5 MCP server."""
    return os.environ.get("MT5_MCP_URL")


def mt5_api_key() -> str | None:
    """Optional — a local MT5 MCP server usually needs none."""
    return os.environ.get("MT5_API_KEY")


def mt5_auth_headers() -> dict[str, str] | None:
    """Auth headers for the MT5 MCP call, or `None` when no key is configured.

    `None` rather than `{}` on purpose: `mcp_client` only builds its own HTTP client when
    there are headers to send, so an unset key means the transport behaves exactly as it
    did before a key was ever an option.
    """
    key = mt5_api_key()
    if not key:
        return None
    value = f"{MT5_AUTH_SCHEME} {key}" if MT5_AUTH_SCHEME else key
    return {MT5_AUTH_HEADER: value}


def postgres_uri() -> str | None:
    """Checkpointer connection string for dev/prod. Tests use InMemorySaver."""
    return os.environ.get("POSTGRES_URI")
