"""Memory Graph: a unified read model over relations that already exist.

The graph's substance is mostly *already stored* — episode_tags is an
episode↔concept bipartite graph, evidence_ids are derived_from edges,
memory_conflicts are pre-shaped contradicts edges.  This package adds the
unified typed read model, a small table for explicitly asserted edges, and
mermaid rendering; it derives everything else on demand.
"""

from brain_memory.graph.links import LinkStore
from brain_memory.graph.ppr import personalized_pagerank
from brain_memory.graph.render import render_mermaid
from brain_memory.graph.view import GraphView, parse_ref

__all__ = [
    "GraphView",
    "LinkStore",
    "parse_ref",
    "render_mermaid",
    "personalized_pagerank",
]
