"""Hybrid retrieval: vector + keyword channels, union, factor scoring, rank.

Design: the two channels are *candidate generators* (recall-oriented); the
normalized multi-factor scorer then ranks the union (precision-oriented).
This is the same shape as the design doc's "Retrieval Engine" and keeps each
channel simple: the vector channel covers paraphrase, the keyword channel
covers exact terms and rare entities that embeddings blur together.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from brain_memory.config import MemoryConfig
from brain_memory.episodic.store import EpisodicStore
from brain_memory.models import Episode, ExtractedExperience, RecallResult
from brain_memory.retrieval import ranking
from brain_memory.retrieval.vector_index import VectorIndex


def build_fts_match_expr(cue: ExtractedExperience) -> str:
    """Build an OR-of-quoted-phrases MATCH expression.

    The index side stores CJK characters space-separated (unicode61 has no CJK
    segmentation), and the query side quotes each run — the query parser then
    produces single-char token phrases, which match the indexed adjacency.
    """
    terms: list[str] = []
    for token in ranking._LATIN_RE.findall(cue.content.lower()):
        if len(token) >= 2:
            terms.append(token)
    for entity in cue.entities:
        for token in ranking._LATIN_RE.findall(entity.lower()):
            if len(token) >= 2:
                terms.append(token)
    quoted = [_quote(term) for term in terms]
    for run in ranking._CJK_RE.findall(cue.content):
        if len(run) >= 2:
            quoted.append(_quote(run))
    # dedup, keep order
    seen: set[str] = set()
    unique = [q for q in quoted if not (q in seen or seen.add(q))]
    return " OR ".join(unique)


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def passes_filters(episode, source: str | None, time_from: datetime | None,
                   time_to: datetime | None, require_entities: list[str] | None) -> bool:
    """Shared recall filter — also applied to graph-expanded candidates so
    expansion can never leak memories the caller filtered out."""
    if not episode.is_active:
        return False
    if source is not None and episode.source != source:
        return False
    if time_from is not None and episode.created_at < time_from:
        return False
    if time_to is not None and episode.created_at > time_to:
        return False
    if require_entities:
        have = {e.casefold() for e in episode.entities}
        wanted = {e.casefold() for e in require_entities}
        if not wanted & have:
            return False
    return True


def semantic_view(memory):
    """Episode-shaped read-only view of a semantic memory (see RecallResult)."""
    return Episode(
        id=memory.id,
        content=memory.statement,
        content_hash=f"semantic-{memory.id}",
        entities=[memory.concept],
        topics=[],
        created_at=memory.updated_at,
        importance=memory.confidence,
        confidence=memory.confidence,
        access_count=memory.access_count,
        last_accessed=memory.last_accessed,
        status=memory.status,
        metadata={"kind": "semantic", "concept": memory.concept},
    )


class Retriever:
    def __init__(
        self,
        store: EpisodicStore,
        index: VectorIndex,
        config: MemoryConfig,
        semantic_store=None,
        semantic_index: VectorIndex | None = None,
    ) -> None:
        self._store = store
        self._index = index
        self._config = config
        self._semantic_store = semantic_store
        self._semantic_index = semantic_index

    def retrieve(
        self,
        cue_variants: list[ExtractedExperience],
        vectors: list[np.ndarray],
        k: int | None = None,
        *,
        source: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        require_entities: list[str] | None = None,
        entity_boost: list[str] | None = None,
    ) -> list[RecallResult]:
        """Rank candidates for one or more cue variants (e.g. original + LLM
        expansion).  Channels union across variants: semantic takes the max
        similarity, keyword the best (lowest) bm25 rank.

        ``entity_boost`` feeds the entity-overlap *factor* only — it never
        becomes a query term.  Session entities bias ranking toward memories
        related to what the agent is currently doing without widening the
        candidate channels (query terms come from the variants alone).
        """
        k = k or self._config.default_top_k
        pool = self._config.candidate_pool_per_channel
        now = datetime.now(timezone.utc)

        candidates: dict[int, tuple[float, float]] = {}  # id -> (best semantic, best bm25 rank)
        for variant, vector in zip(cue_variants, vectors):
            for episode_id, similarity in self._index.search(vector, pool):
                best_sim, best_rank = candidates.get(episode_id, (-1.0, math.inf))
                candidates[episode_id] = (max(best_sim, similarity), best_rank)
            match_expr = build_fts_match_expr(variant)
            if match_expr:
                for episode, rank in self._store.fts_search(match_expr, pool):
                    best_sim, best_rank = candidates.get(episode.id, (-1.0, math.inf))
                    candidates[episode.id] = (best_sim, min(best_rank, rank))
        if not candidates:
            return []

        cue_entities: set[str] = set()
        cue_context: set[str] = set()
        for variant in cue_variants:
            cue_entities |= {e.casefold() for e in variant.entities}
            cue_context |= {t.casefold() for t in variant.topics}
            if variant.context:
                cue_context |= ranking._context_tokens(variant.context)
        if entity_boost:
            cue_entities |= {e.casefold() for e in entity_boost}
        cue_entities = set(sorted(cue_entities)[:12])

        episodic = self._rank_episodic(
            candidates, cue_entities, cue_context, now,
            source=source, time_from=time_from, time_to=time_to,
            require_entities=require_entities,
        )
        if self._semantic_store is None or self._semantic_index is None:
            episodic.sort(key=lambda result: -result.score)
            return episodic[:k]

        semantic = self._rank_semantic(
            cue_variants, vectors, cue_entities, cue_context, now, pool
        )
        merged = self._merge_with_cap(episodic, semantic)
        return merged[:k]

    # -- episodic channel ---------------------------------------------------------

    def _rank_episodic(
        self,
        candidates: dict[int, tuple[float, float]],
        cue_entities: set[str],
        cue_context: set[str],
        now: datetime,
        *,
        source: str | None,
        time_from: datetime | None,
        time_to: datetime | None,
        require_entities: list[str] | None,
    ) -> list[RecallResult]:
        episodes = self._store.get_many(list(candidates.keys()))
        episodes = [
            episode
            for episode in episodes
            if passes_filters(episode, source, time_from, time_to, require_entities)
        ]
        results: list[RecallResult] = []
        for episode in episodes:
            semantic_sim, bm25_rank = candidates[episode.id]
            keyword = ranking.normalize_bm25(bm25_rank) if bm25_rank < math.inf else 0.0
            factors = ranking.compute_factors(
                episode=episode,
                semantic=semantic_sim,
                keyword=keyword,
                cue_entities=cue_entities,
                cue_context=cue_context,
                now=now,
                half_life_days=self._config.recency_half_life_days,
            )
            score = ranking.score(factors, self._config.weights)
            results.append(
                RecallResult(
                    episode=episode,
                    kind="episodic",
                    score=score,
                    factors=factors,
                    reasons=ranking.explain(factors, self._config.weights),
                )
            )
        return results

    # -- semantic channel ------------------------------------------------------------

    def _rank_semantic(
        self,
        cue_variants: list[ExtractedExperience],
        vectors: list[np.ndarray],
        cue_entities: set[str],
        cue_context: set[str],
        now: datetime,
        pool: int,
    ) -> list[RecallResult]:
        candidates: dict[int, tuple[float, float]] = {}
        for variant, vector in zip(cue_variants, vectors):
            for semantic_id, similarity in self._semantic_index.search(vector, pool):
                best_sim, best_rank = candidates.get(semantic_id, (-1.0, math.inf))
                candidates[semantic_id] = (max(best_sim, similarity), best_rank)
            match_expr = build_fts_match_expr(variant)
            if match_expr:
                for memory, rank in self._semantic_store.fts_search(match_expr, pool):
                    best_sim, best_rank = candidates.get(memory.id, (-1.0, math.inf))
                    candidates[memory.id] = (best_sim, min(best_rank, rank))
        if not candidates:
            return []

        memories = self._semantic_store.get_many(list(candidates.keys()))
        results: list[RecallResult] = []
        for memory in memories:
            if not memory.status.value == "active":
                continue
            view = semantic_view(memory)
            semantic_sim, bm25_rank = candidates[memory.id]
            keyword = ranking.normalize_bm25(bm25_rank) if bm25_rank < math.inf else 0.0
            factors = ranking.compute_factors(
                episode=view,
                semantic=semantic_sim,
                keyword=keyword,
                cue_entities=cue_entities,
                cue_context=cue_context,
                now=now,
                half_life_days=self._config.recency_half_life_days,
            )
            score = ranking.score(factors, self._config.weights)
            results.append(
                RecallResult(
                    episode=view,
                    kind="semantic",
                    semantic=memory,
                    score=score,
                    factors=factors,
                    reasons=ranking.explain(factors, self._config.weights),
                )
            )
        return results

    def _merge_with_cap(
        self, episodic: list[RecallResult], semantic: list[RecallResult]
    ) -> list[RecallResult]:
        """Interleave both channels by score, but never let semantic memories
        flood the recall list (``semantic_recall_limit``)."""
        merged = [*episodic, *semantic]
        merged.sort(key=lambda result: -result.score)
        kept: list[RecallResult] = []
        semantic_kept = 0
        for result in merged:
            if result.is_semantic:
                if semantic_kept >= self._config.semantic_recall_limit:
                    continue
                semantic_kept += 1
            kept.append(result)
        return kept

