"""Text rendering of graph subgraphs (mermaid), feeding the future Inspector."""

from __future__ import annotations

from brain_memory.models import GraphSubgraph


def _escape(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def render_mermaid(subgraph: GraphSubgraph) -> str:
    lines = ["graph TD"]
    ids = {node.ref: f"n{i}" for i, node in enumerate(subgraph.nodes)}
    for node in subgraph.nodes:
        shape = f'{ids[node.ref]}["{_escape(node.label)}"]'
        if node.status is not None and node.status.value != "active":
            shape = f'{ids[node.ref]}[/"{_escape(node.label)}"/]'
        lines.append(f"    {shape}")
    for edge in subgraph.edges:
        source, target = ids.get(edge.source), ids.get(edge.target)
        if source is None or target is None:
            continue
        lines.append(f"    {source} -->|{edge.kind.value}| {target}")
    return "\n".join(lines)
