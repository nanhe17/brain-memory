"""图谱子图的文本渲染（mermaid），为将来的 Inspector 提供基础。"""

from __future__ import annotations

from brain_memory.models import GraphSubgraph


def _escape(text: str) -> str:
    """mermaid 标签内的字符转义。"""
    return text.replace('"', "'").replace("\n", " ")


def render_mermaid(subgraph: GraphSubgraph) -> str:
    """子图 -> mermaid TD 文本（非活跃节点用斜框表示）。"""
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
