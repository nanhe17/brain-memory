"""The typed graph read model.

`neighborhood()` merges four edge sources into one subgraph:

* ``mentions``      episode -> concept            (episode_tags, provenance tags)
* ``derived_from``  semantic -> episode           (evidence_ids, provenance evidence)
* ``contradicts``   semantic -> episode           (memory_conflicts, provenance conflict)
* ``co_occurs_with`` concept <-> concept          (tags self-join, provenance co_occurrence)
* ``similar_to``    episode <-> episode           (vector kNN on demand, provenance vector)
* explicit relations from ``memory_links``        (provenance explicit)

Lifecycle: edges whose endpoints have left the active state are filtered at
read time — no cascade deletes, so `restore()` brings relationships back by
itself.  Concept nodes are always considered alive.
"""

from __future__ import annotations

from brain_memory.consolidation.conflicts import ConflictStore
from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.episodic.store import EpisodicStore
from brain_memory.graph.links import LinkStore
from brain_memory.models import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    MemoryStatus,
    NodeKind,
)
from brain_memory.retrieval.vector_index import VectorIndex
from brain_memory.storage.db import Database

_MAX_LABEL = 60


def parse_ref(ref: str) -> tuple[NodeKind, int | None, str | None]:
    """'e12' -> (episode, 12, None); 's3' -> (semantic, 3, None);
    'c:java' -> (concept, None, 'java')."""
    ref = (ref or "").strip()
    if ref.startswith("c:"):
        value = ref[2:].strip().casefold()
        if not value:
            raise ValueError(f"empty concept ref: {ref!r}")
        return NodeKind.CONCEPT, None, value
    if ref.startswith("e"):
        try:
            return NodeKind.EPISODE, int(ref[1:]), None
        except ValueError:
            raise ValueError(f"bad episode ref: {ref!r}") from None
    if ref.startswith("s"):
        try:
            return NodeKind.SEMANTIC, int(ref[1:]), None
        except ValueError:
            raise ValueError(f"bad semantic ref: {ref!r}") from None
    raise ValueError(f"unrecognized ref {ref!r}: use 'e<id>', 's<id>' or 'c:<concept>'")


def episode_ref(episode_id: int) -> str:
    return f"e{episode_id}"


def semantic_ref(semantic_id: int) -> str:
    return f"s{semantic_id}"


def concept_ref(value: str) -> str:
    return f"c:{value.casefold()}"


class GraphView:
    def __init__(
        self,
        db: Database,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        conflict_store: ConflictStore,
        link_store: LinkStore,
        episodic_index: VectorIndex | None = None,
    ) -> None:
        self._db = db
        self._episodic = episodic_store
        self._semantic = semantic_store
        self._conflicts = conflict_store
        self._links = link_store
        self._index = episodic_index

    # -- public ----------------------------------------------------------------

    def neighborhood(
        self,
        ref: str,
        *,
        include_similar: bool = False,
        max_per_kind: int = 6,
    ) -> GraphSubgraph:
        kind, node_id, concept = parse_ref(ref)
        sub = GraphSubgraph(center=ref)
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        def add_node(node: GraphNode) -> str:
            nodes.setdefault(node.ref, node)
            return node.ref

        def add_edge(edge: GraphEdge) -> None:
            edges.append(edge)

        if kind is NodeKind.EPISODE:
            episode = self._episodic.get(node_id, with_embedding=True)
            if episode is None:
                raise ValueError(f"episode {node_id} not found")
            add_node(self._episode_node(episode))
            for value in self._episode_concepts(episode)[:max_per_kind]:
                concept_node = self._concept_node(value)
                add_node(concept_node)
                add_edge(GraphEdge(source=episode_ref(node_id), target=concept_node.ref,
                                   kind=EdgeKind.MENTIONS, provenance="tags"))
            self._add_explicit(add_node, add_edge, NodeKind.EPISODE, node_id)
            if include_similar and self._index is not None and episode.embedding is not None:
                self._add_similar(add_node, add_edge, episode, max_per_kind)

        elif kind is NodeKind.SEMANTIC:
            memory = self._semantic.get(node_id)
            if memory is None:
                raise ValueError(f"semantic memory {node_id} not found")
            add_node(self._semantic_node(memory))
            for episode in self._active_episodes(memory.evidence_ids)[:max_per_kind]:
                add_node(self._episode_node(episode))
                add_edge(GraphEdge(source=semantic_ref(node_id),
                                   target=episode_ref(episode.id),
                                   kind=EdgeKind.DERIVED_FROM, provenance="evidence"))
            for conflict in self._conflicts.for_semantic(node_id):
                if conflict.trigger_episode_id is None:
                    continue
                episode = self._episodic.get(conflict.trigger_episode_id)
                if episode is None or not episode.is_active:
                    continue
                add_node(self._episode_node(episode))
                add_edge(GraphEdge(source=semantic_ref(node_id),
                                   target=episode_ref(episode.id),
                                   kind=EdgeKind.CONTRADICTS, provenance="conflict",
                                   weight=1.0))
            self._add_explicit(add_node, add_edge, NodeKind.SEMANTIC, node_id)

        else:
            concept_node = self._concept_node(concept)
            add_node(concept_node)
            ids = self._concept_episode_ids(concept)
            for episode in self._active_episodes(ids)[:max_per_kind]:
                add_node(self._episode_node(episode))
                add_edge(GraphEdge(source=episode_ref(episode.id),
                                   target=concept_node.ref,
                                   kind=EdgeKind.MENTIONS, provenance="tags"))
            for row in self._db.concept_co_occurrence(concept, max_per_kind):
                other = self._concept_node(row["other"])
                add_node(other)
                add_edge(GraphEdge(source=concept_node.ref, target=other.ref,
                                   kind=EdgeKind.CO_OCCURS_WITH,
                                   weight=float(row["n"]), provenance="co_occurrence"))
            memory = self._semantic.get_by_concept(concept)
            if memory is not None and memory.status is MemoryStatus.ACTIVE:
                add_node(self._semantic_node(memory))
                add_edge(GraphEdge(source=concept_node.ref,
                                   target=semantic_ref(memory.id),
                                   kind=EdgeKind.RELATED_TO, provenance="consolidation",
                                   weight=memory.confidence))

        sub.nodes = list(nodes.values())
        sub.edges = edges
        return sub

    def related_concepts(self, concept: str, *, limit: int = 5) -> list[tuple[str, int]]:
        return [
            (row["other"], int(row["n"]))
            for row in self._db.concept_co_occurrence(concept.casefold(), limit)
        ]

    # -- helpers -----------------------------------------------------------------

    def _add_explicit(self, add_node, add_edge, kind: NodeKind, node_id: int) -> None:
        for link in self._links.for_node(kind, node_id):
            source_ref = self._instance_ref(link.source_kind, link.source_id)
            target_ref = self._instance_ref(link.target_kind, link.target_id)
            source_episode = (
                self._episodic.get(link.source_id) if link.source_kind is NodeKind.EPISODE else None
            )
            source_semantic = (
                self._semantic.get(link.source_id)
                if link.source_kind is NodeKind.SEMANTIC
                else None
            )
            target_episode = (
                self._episodic.get(link.target_id) if link.target_kind is NodeKind.EPISODE else None
            )
            target_semantic = (
                self._semantic.get(link.target_id)
                if link.target_kind is NodeKind.SEMANTIC
                else None
            )
            if source_episode is not None:
                node = self._episode_node(source_episode)
            elif source_semantic is not None:
                node = self._semantic_node(source_semantic)
            else:
                continue
            if target_episode is not None:
                other = self._episode_node(target_episode)
            elif target_semantic is not None:
                other = self._semantic_node(target_semantic)
            else:
                continue
            # lifecycle: an endpoint that left the active state hides the edge
            if (source_episode and not source_episode.is_active) or (
                source_semantic and source_semantic.status is not MemoryStatus.ACTIVE
            ):
                continue
            if (target_episode and not target_episode.is_active) or (
                target_semantic and target_semantic.status is not MemoryStatus.ACTIVE
            ):
                continue
            add_node(node)
            add_node(other)
            add_edge(GraphEdge(source=node.ref, target=other.ref,
                               kind=EdgeKind(link.relation), weight=link.weight,
                               provenance="explicit"))

    def _add_similar(self, add_node, add_edge, episode, max_per_kind: int) -> None:
        for similar_id, similarity in self._index.search(episode.embedding, max_per_kind + 1):
            if similar_id == episode.id:
                continue
            other = self._episodic.get(similar_id)
            if other is None or not other.is_active:
                continue
            add_node(self._episode_node(other))
            add_edge(GraphEdge(source=episode_ref(episode.id),
                               target=episode_ref(similar_id),
                               kind=EdgeKind.SIMILAR_TO, weight=round(similarity, 4),
                               provenance="vector"))

    @staticmethod
    def _instance_ref(kind: NodeKind, node_id: int) -> str:
        return episode_ref(node_id) if kind is NodeKind.EPISODE else semantic_ref(node_id)

    def _episode_node(self, episode) -> GraphNode:
        return GraphNode(
            ref=episode_ref(episode.id),
            kind=NodeKind.EPISODE,
            id=episode.id,
            label=f"#{episode.id} {episode.content[:_MAX_LABEL]}",
            status=episode.status,
            detail=episode.content,
        )

    def _semantic_node(self, memory) -> GraphNode:
        return GraphNode(
            ref=semantic_ref(memory.id),
            kind=NodeKind.SEMANTIC,
            id=memory.id,
            label=f"S-{memory.id} [{memory.concept}] {memory.statement[:_MAX_LABEL]}",
            status=memory.status,
            detail=memory.statement,
        )

    @staticmethod
    def _concept_node(value: str) -> GraphNode:
        return GraphNode(ref=concept_ref(value), kind=NodeKind.CONCEPT, label=value)

    @staticmethod
    def _episode_concepts(episode) -> list[str]:
        seen: dict[str, None] = {}
        for tag in [*episode.entities, *episode.topics]:
            seen.setdefault(tag.casefold())
        return list(seen)

    def _concept_episode_ids(self, concept: str) -> list[int]:
        ids = set(self._db.episode_ids_by_tag("entity", concept))
        ids.update(self._db.episode_ids_by_tag("topic", concept))
        return sorted(ids)

    def _active_episodes(self, episode_ids: list[int]):
        episodes = self._episodic.get_many(episode_ids)
        return [e for e in episodes if e.is_active]
