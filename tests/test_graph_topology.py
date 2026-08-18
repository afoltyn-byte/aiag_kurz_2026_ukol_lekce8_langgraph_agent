"""Topology drift guard.

Compares the hand-maintained spec diagram (docs/graph.mmd) against the Mermaid
that LangGraph emits from the *compiled* graph. Any node or edge that exists on
one side and not the other fails the test.

This is the whole point of the diagram-first workflow: the drawing cannot quietly
stop describing the code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIAGRAM = next(
    (p for p in (REPO_ROOT / "docs" / "graph.mmd", REPO_ROOT / "docs" / "graph.md") if p.exists()),
    REPO_ROOT / "docs" / "graph.mmd",
)

# ids used in the spec diagram -> ids LangGraph emits
SENTINEL_ALIASES = {
    "START": "__start__",
    "END": "__end__",
    "__start__": "__start__",
    "__end__": "__end__",
}

_SKIP_PREFIXES = (
    "%%",
    "flowchart",
    "graph",
    "subgraph",
    "end",
    "direction",
    "classDef",
    "class ",
    "style",
    "linkStyle",
    "click",
    "title",
)

_ID = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)")

# a standalone node declaration and nothing else, e.g.  analyze(analyze):::first
_DECL = re.compile(
    r"^(?P<id>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?:\[.*\]|\(.*\)|\{.*\}|>.*\])?"
    r"\s*(?::::[A-Za-z0-9_]+)?\s*;?$"
)


def _extract_mermaid_block(source: str) -> str:
    """If the file is Markdown, use only the first ```mermaid fenced block.

    Prose in a .md file can otherwise be mistaken for standalone node
    declarations. A bare .mmd file is returned unchanged.
    """
    lines = source.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("```") and "mermaid" in line:
            for close in range(idx + 1, len(lines)):
                if lines[close].strip().startswith("```"):
                    return "\n".join(lines[idx + 1 : close])
            return "\n".join(lines[idx + 1 :])
    return source


def _strip_frontmatter(source: str) -> str:
    """LangGraph prefixes draw_mermaid() output with a YAML config block."""
    lines = source.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :])
    return source


def _clean_label(label: str) -> str:
    label = label.replace("&nbsp;", " ")
    label = re.sub(r"</?[a-zA-Z][^>]*>", "", label)
    return label.strip()


def _normalise_arrows(line: str) -> str:
    """Collapse every Mermaid link flavour into `-->` or `-->|label|`."""
    # dotted link with inline label:  A -. label .-> B
    line = re.sub(r"-\.\s*([^.]*?)\s*\.-+>", r"-->|\1|", line)
    # bare dotted / thick / long links
    line = re.sub(r"-\.-*>", "-->", line)
    line = re.sub(r"=+>", "-->", line)
    line = re.sub(r"-{3,}>", "-->", line)
    # thick link with label:  A == label ==> B  (rare, normalise defensively)
    line = re.sub(r"=+\s*([^=]*?)\s*=+>", r"-->|\1|", line)
    return line


def _strip_shape(fragment: str) -> str:
    """Drop everything from the first shape opener onwards."""
    return re.split(r"[\[({>]", fragment, maxsplit=1)[0]


def _tail_id(fragment: str) -> str | None:
    cleaned = _strip_shape(fragment).strip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", cleaned)
    return match.group(1) if match else None


def _head_label_and_id(fragment: str) -> tuple[str | None, str | None]:
    fragment = fragment.lstrip()
    label = None
    label_match = re.match(r"\|([^|]*)\|", fragment)
    if label_match:
        label = _clean_label(label_match.group(1))
        fragment = fragment[label_match.end() :].lstrip()
    match = _ID.match(fragment)
    return label, (match.group(1) if match else None)


def _alias(node_id: str) -> str:
    return SENTINEL_ALIASES.get(node_id, node_id)


def parse_mermaid_flowchart(source: str) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (nodes, edges) as sets. Edge labels are deliberately ignored here;
    they are checked separately so a label typo does not read as a missing edge.
    """
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()

    for raw in _strip_frontmatter(_extract_mermaid_block(source)).splitlines():
        line = raw.strip().rstrip(";")
        if not line or line.startswith(_SKIP_PREFIXES):
            continue

        line = _normalise_arrows(line)

        if "-->" not in line:
            declaration = _DECL.match(line)
            if declaration:
                nodes.add(_alias(declaration.group("id")))
            continue

        parts = line.split("-->")
        for left, right in zip(parts, parts[1:]):
            src = _tail_id(left)
            _, dst = _head_label_and_id(right)
            if src and dst:
                src, dst = _alias(src), _alias(dst)
                nodes.update({src, dst})
                edges.add((src, dst))

    return nodes, edges


def parse_mermaid_labels(source: str) -> set[tuple[str, str, str]]:
    """(src, dst, label) for labelled edges only."""
    labelled: set[tuple[str, str, str]] = set()
    for raw in _strip_frontmatter(_extract_mermaid_block(source)).splitlines():
        line = raw.strip().rstrip(";")
        if not line or line.startswith(_SKIP_PREFIXES):
            continue
        line = _normalise_arrows(line)
        if "-->" not in line:
            continue
        parts = line.split("-->")
        for left, right in zip(parts, parts[1:]):
            src = _tail_id(left)
            label, dst = _head_label_and_id(right)
            if src and dst and label:
                labelled.add((_alias(src), _alias(dst), _clean_label(label)))
    return labelled


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec_source() -> str:
    assert SPEC_DIAGRAM.exists(), f"missing spec diagram: {SPEC_DIAGRAM}"
    return SPEC_DIAGRAM.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def implementation_source() -> str:
    pytest.importorskip("langgraph")
    from src.agent.graph import compile_graph

    return compile_graph().get_graph().draw_mermaid()


@pytest.fixture(scope="module")
def implementation_labels() -> set[tuple[str, str, str]]:
    """Conditional-edge labels read from the builder, NOT from draw_mermaid().

    draw_mermaid() is lossy for labels: it omits the label when the mapping key
    equals the target node name, and it collapses several branches that share a
    target into a single drawn edge. StateGraph.branches[node][router].ends is
    the authoritative mapping.
    """
    pytest.importorskip("langgraph")
    from src.agent.graph import build_graph

    labels: set[tuple[str, str, str]] = set()
    for node, branches in build_graph().branches.items():
        for spec in branches.values():
            ends = getattr(spec, "ends", None) or {}
            for label, target in ends.items():
                labels.add((_alias(node), _alias(target), label))
    return labels


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_nodes_match(spec_source: str, implementation_source: str) -> None:
    spec_nodes, _ = parse_mermaid_flowchart(spec_source)
    impl_nodes, _ = parse_mermaid_flowchart(implementation_source)

    only_in_spec = sorted(spec_nodes - impl_nodes)
    only_in_impl = sorted(impl_nodes - spec_nodes)

    assert not only_in_spec, f"drawn but not implemented: {only_in_spec}"
    assert not only_in_impl, f"implemented but not drawn: {only_in_impl}"


def test_edges_match(spec_source: str, implementation_source: str) -> None:
    _, spec_edges = parse_mermaid_flowchart(spec_source)
    _, impl_edges = parse_mermaid_flowchart(implementation_source)

    only_in_spec = sorted(spec_edges - impl_edges)
    only_in_impl = sorted(impl_edges - spec_edges)

    assert not only_in_spec, f"drawn but not wired: {only_in_spec}"
    assert not only_in_impl, f"wired but not drawn: {only_in_impl}"


def test_conditional_edge_labels_match(
    spec_source: str, implementation_labels: set[tuple[str, str, str]]
) -> None:
    """Router return values must equal the labels on the drawing.

    A mismatch here is the single most common LangGraph bug: the mapping key and
    the router's return string drift apart and the branch silently never fires.
    """
    spec_labels = parse_mermaid_labels(spec_source)
    impl_labels = implementation_labels

    only_in_spec = sorted(spec_labels - impl_labels)
    only_in_impl = sorted(impl_labels - spec_labels)

    assert not only_in_spec, f"labels drawn but not produced by routers: {only_in_spec}"
    assert not only_in_impl, f"labels in code but not drawn: {only_in_impl}"


def test_no_orphan_nodes(implementation_source: str) -> None:
    nodes, edges = parse_mermaid_flowchart(implementation_source)
    reachable = {dst for _, dst in edges} | {"__start__"}
    outgoing = {src for src, _ in edges} | {"__end__"}

    unreachable = sorted(nodes - reachable)
    dead_end = sorted(nodes - outgoing)

    assert not unreachable, f"unreachable nodes: {unreachable}"
    assert not dead_end, f"nodes with no outgoing edge: {dead_end}"
