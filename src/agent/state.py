"""State schema for the instrument-analysis graph.

Contract: `docs/graph-spec.md` §3. Every key here appears in that table, and every
key in that table appears here. A key with no reducer is a silent data-loss bug the
moment two nodes write it in the same superstep, so the two append-only keys carry
`operator.add` and `messages` carries `add_messages`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Scope = Literal["news", "chart", "both"]
"""What the user asked for. `None` on state means 'not yet resolved'."""

NextAgent = Literal["clarify", "analytics", "trader", "writer", "done"]
"""The supervisor's *proposal*. `route_from_supervisor` may override it."""

RouteLabel = Literal["clarify", "analytics", "trader", "writer", "done", "step_limit"]
"""Router return values == conditional-edge labels in `docs/graph.md`.

These two must stay identical: a drifted label means the branch silently never
fires. `tests/test_graph_topology.py::test_conditional_edge_labels_match` is the
guard.
"""

VALID_SCOPES: tuple[Scope, ...] = ("news", "chart", "both")


class AgentState(TypedDict, total=False):
    """Full graph state. Nodes return *partial* dicts of these keys, never all."""

    # --- input ---
    request: str  # raw user text, never mutated

    # --- trace ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- resolved by supervisor ---
    instrument: str | None  # broker symbol as it exists on MT5, uppercase
    timeframe: str  # MT5 timeframe, defaults to config.DEFAULT_TIMEFRAME
    language: str | None  # ISO 639-1, detected once from `request`
    scope: Scope | None
    next_agent: NextAgent | None
    step_count: int  # +1 per supervisor visit

    # --- written by clarify ---
    clarify_count: int  # +1 per question asked

    # --- artifacts ---
    analytics_result: dict[str, Any] | None
    chart: dict[str, Any] | None  # holds a path, never image bytes
    document: dict[str, Any] | None

    # --- append-only diagnostics ---
    errors: Annotated[list[str], operator.add]
    agent_log: Annotated[list[dict[str, Any]], operator.add]


class InputState(TypedDict):
    """Public input surface: `request` only, and it is required."""

    request: str


class OutputState(TypedDict, total=False):
    """Public output surface, per spec §3 'Output keys'."""

    document: dict[str, Any] | None
    chart: dict[str, Any] | None
    analytics_result: dict[str, Any] | None
    scope: Scope | None
    errors: list[str]
    agent_log: list[dict[str, Any]]
