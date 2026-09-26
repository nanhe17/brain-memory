"""PersonalizedPageRank over the active memory subgraph (research extension).

Seeds are concept nodes; mass diffuses along mentions (episode<->concept),
evidence (semantic<->episode, weighted by confidence) and consolidation
(concept->its semantic) edges.  Recomputed on demand — no incremental
updates, which is the right trade at the scales this engine targets
(O(E) per round, pure Python, tens of milliseconds at 10k memories).

Node refs follow the graph conventions: 'e<id>', 's<id>', 'c:<concept>'.
"""

from __future__ import annotations

from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.episodic.store import EpisodicStore
from brain_memory.storage.db import Database


def _seed_distribution(seeds: dict[str, float]) -> dict[str, float]:
    total = sum(seeds.values())
    if total <= 0:
        return {}
    return {ref: weight / total for ref, weight in seeds.items()}


def personalized_pagerank(
    db: Database,
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    *,
    seeds: dict[str, float],
    damping: float = 0.85,
    iterations: int = 30,
    tol: float = 1e-6,
    max_nodes: int = 20000,
) -> dict[str, float]:
    """PPR mass per node ref.  Deterministic; mass sums to ~1.

    Edges are traversed in both directions; each node distributes its
    outflowing mass proportionally to edge weight, and dangling mass (nodes
    without neighbors) teleports back to the seed distribution.
    """
    seed_dist = _seed_distribution(seeds)
    if not seed_dist:
        return {}

    adjacency, node_refs = _build_graph(db, episodic_store, semantic_store, max_nodes)
    seeds = {ref: weight for ref, weight in seeds.items() if ref in adjacency}
    seed_dist = _seed_distribution(seeds)
    if not seed_dist:
        return {}

    mass = dict(seed_dist)
    for _ in range(iterations):
        nxt = {ref: (1.0 - damping) * seed_dist.get(ref, 0.0) for ref in adjacency}
        dangling = 0.0
        for ref, m in mass.items():
            neighbors = adjacency[ref]
            if not neighbors:
                dangling += m
                continue
            total_weight = sum(neighbors.values())
            for other, weight in neighbors.items():
                nxt[other] += damping * m * (weight / total_weight)
        if dangling > 0:
            for ref, share in seed_dist.items():
                nxt[ref] += damping * dangling * share
        delta = sum(abs(nxt[ref] - mass.get(ref, 0.0)) for ref in nxt)
        mass = nxt
        if delta < tol:
            break
    return mass


def _build_graph(db: Database, episodic_store: EpisodicStore,
                 semantic_store: SemanticStore, max_nodes: int):
    """Sparse adjacency over the active subgraph (bidirectional edges)."""
    adjacency: dict[str, dict[str, float]] = {}

    def link(a: str, b: str, weight: float) -> None:
        adjacency.setdefault(a, {})[b] = max(weight, adjacency.get(a, {}).get(b, 0.0))
        adjacency.setdefault(b, {})[a] = max(weight, adjacency.get(b, {}).get(a, 0.0))

    semantics = {m.id: m for m in semantic_store.list_active()}
    episodes = episodic_store.all_active()
    if len(episodes) + len(semantics) > max_nodes:
        episodes = episodes[: max_nodes - len(semantics)]

    for episode in episodes:
        ref = f"e{episode.id}"
        adjacency.setdefault(ref, {})
        for value in {t.casefold() for t in [*episode.entities, *episode.topics]}:
            link(ref, f"c:{value}", 1.0)

    for memory in semantics.values():
        sref = f"s{memory.id}"
        adjacency.setdefault(sref, {})
        for episode_id in memory.evidence_ids:
            episode = episodic_store.get(episode_id)
            if episode is not None and episode.is_active:
                link(sref, f"e{episode_id}", max(memory.confidence, 0.05))
        link(f"c:{memory.concept}", sref, max(memory.confidence, 0.05))

    node_refs = set(adjacency)
    # drop edges pointing at nodes that were cut by max_nodes
    for ref, neighbors in adjacency.items():
        adjacency[ref] = {n: w for n, w in neighbors.items() if n in node_refs}
    return adjacency, node_refs
