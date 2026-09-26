"""Server-side SVG rendering of graph subgraphs.

Deliberately dependency-free (no mermaid.js, no CDN): the Inspector must
work fully offline.  Layout is a simple two-column scheme — center node on
the left, its neighbors stacked on the right, edges drawn as straight lines
with their kind as the label.  Deterministic output for a given subgraph.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from brain_memory.models import GraphSubgraph

_BOX_W, _BOX_H, _GAP_Y = 260, 44, 18
_CENTER_X, _NEIGHBOR_X = 20, 380
_MARGIN = 16


def _escape(text: str) -> str:
    """XML-escape including quotes (saxutils.escape leaves them alone)."""
    return escape(text, {'"': "&quot;"}).replace("\n", " ")


def subgraph_to_svg(subgraph: GraphSubgraph) -> str:
    nodes = {n.ref: n for n in subgraph.nodes}
    neighbors = [n for n in subgraph.nodes if n.ref != subgraph.center]
    height = max(_BOX_H, len(neighbors) * (_BOX_H + _GAP_Y) + _MARGIN * 2)
    width = _NEIGHBOR_X + _BOX_W + _MARGIN * 2

    def box(x: float, y: float, node) -> str:
        color = "#dbeafe" if node.kind.value == "episode" else (
            "#dcfce7" if node.kind.value == "semantic" else "#fef9c3")
        label = _escape(node.label[:34])
        status = "" if node.status is None or node.status.value == "active" \
            else f" [{node.status.value}]"
        return (
            f'<rect x="{x}" y="{y}" width="{_BOX_W}" height="{_BOX_H}" rx="8" '
            f'fill="{color}" stroke="#64748b"/>\n'
            f'<text x="{x + 10}" y="{y + 27}" font-size="12" '
            f'font-family="monospace">{label}{_escape(status)}</text>'
        )

    center_y = max(_MARGIN, (height - _BOX_H) / 2)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        box(_CENTER_X, center_y, nodes[subgraph.center]),
    ]
    neighbor_y = {}
    for index, node in enumerate(neighbors):
        y = _MARGIN + index * (_BOX_H + _GAP_Y)
        neighbor_y[node.ref] = y
        parts.append(box(_NEIGHBOR_X, y, node))

    positions = {subgraph.center: (_CENTER_X, center_y)}
    positions.update({ref: (_NEIGHBOR_X, y) for ref, y in neighbor_y.items()})
    for edge in subgraph.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        sx = positions[edge.source][0] + _BOX_W if positions[edge.source][0] < _NEIGHBOR_X \
            else positions[edge.source][0]
        sy = positions[edge.source][1] + _BOX_H / 2
        tx = positions[edge.target][0] if positions[edge.target][0] > _CENTER_X + _BOX_W \
            else positions[edge.target][0] + _BOX_W
        ty = positions[edge.target][1] + _BOX_H / 2
        parts.append(
            f'<line x1="{sx}" y1="{sy}" x2="{tx}" y2="{ty}" stroke="#94a3b8" '
            f'marker-end="url(#arrow)"/>'
        )
        mid_x, mid_y = (sx + tx) / 2, (sy + ty) / 2 - 6
        parts.append(
            f'<text x="{mid_x}" y="{mid_y}" font-size="10" fill="#475569" '
            f'text-anchor="middle">{_escape(edge.kind.value)}</text>'
        )

    parts.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
