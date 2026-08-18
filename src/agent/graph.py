"""Graph assembly. Wiring only — no prompts, no I/O, no branching.

Topology is defined by `docs/graph.md`; the edge contracts by `docs/graph-spec.md`
§5. `tests/test_graph_topology.py` compares this wiring against the diagram, so the
two cannot quietly drift.

The keys of `ROUTE_MAP` are the router's return values *and* the edge labels on the
diagram. They must be identical strings — a drifted key means the branch never
fires, silently.
"""

from __future__ import annotations

from collections.abc import Hashable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent import clarify, supervisor
from src.agent.state import AgentState, InputState, OutputState
from src.agents import analytics, trader, writer

# Typed `Hashable` to match `add_conditional_edges`: dict is invariant in its key
# type, so a `dict[str, str]` is not accepted as a `dict[Hashable, str]`.
ROUTE_MAP: dict[Hashable, str] = {
    "clarify": "clarify",
    "analytics": "analytics",
    "trader": "trader",
    "writer": "writer",
    "done": END,
    "step_limit": END,
}


def build_graph() -> StateGraph:
    """The uncompiled builder — `.branches` is what the label drift guard reads."""
    builder = StateGraph(
        AgentState,
        input_schema=InputState,
        output_schema=OutputState,
    )

    builder.add_node("supervisor", supervisor.run)
    builder.add_node("clarify", clarify.run)
    builder.add_node("analytics", analytics.run)
    builder.add_node("trader", trader.run)
    builder.add_node("writer", writer.run)

    builder.add_edge(START, "supervisor")

    # The supervisor is the only branching node. Explicit path_map, never a list.
    builder.add_conditional_edges("supervisor", supervisor.route_from_supervisor, ROUTE_MAP)

    # Every agent hands control straight back to the supervisor.
    builder.add_edge("clarify", "supervisor")
    builder.add_edge("analytics", "supervisor")
    builder.add_edge("trader", "supervisor")
    builder.add_edge("writer", "supervisor")

    return builder


def compile_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compile with a checkpointer — required, not optional: `clarify` interrupts,
    and an interrupted run cannot resume without one.

    Defaults to `InMemorySaver` so tests and the topology guard need no database.
    Dev/prod passes a `PostgresSaver` built from `config.postgres_uri()`.
    """
    return build_graph().compile(checkpointer=checkpointer or InMemorySaver())
