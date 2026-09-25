"""Hybrid retrieval: vector + keyword channels, union, factor scoring, rank.

Design: the two channels are *candidate generators* (recall-oriented); the
normalized multi-factor scorer then ranks the union (precision-oriented).
This is the same shape as the design doc's "Retrieval Engine" and keeps each
channel simple: the vector channel covers paraphrase, the keyword channel
covers exact terms and rare entities that embeddings blur together.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from brain_memory.config import MemoryConfig
from brain_memory.episodic.store import EpisodicStore
from brain_memory.models import ExtractedExperience, FactorScores, RecallResult
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


class Retriever:
    def __init__(self, store: EpisodicStore, index: VectorIndex, config: MemoryConfig) -> None:
        self._store = store
        self._index = index
        self._config = config

    def retrieve(
        self,
        cue: ExtractedExperience,
        cue_embedding: np.ndarray,
        k: int | None = None,
        *,
        source: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        require_entities: list[str] | None = None,
    ) -> list[RecallResult]:
        k = k or self._config.default_top_k
        pool = self._config.candidate_pool_per_channel
        now = datetime.now(timezone.utc)

        candidates: dict[int, tuple[float, float]] = {}  # id -> (semantic, bm25_rank)
        for episode_id, similarity in self._index.search(cue_embedding, pool):
            candidates[episode_id] = (similarity, -1.0)
        match_expr = build_fts_match_expr(cue)
        if match_expr:
            for episode, rank in self._store.fts_search(match_expr, pool):
                semantic, _ = candidates.get(episode.id, (0.0, -1.0))
                candidates[episode.id] = (semantic, rank)
        if not candidates:
            return []

        episodes = self._store.get_many(list(candidates.keys()))
        episodes = [
            episode
            for episode in episodes
            if self._passes_filters(episode, source, time_from, time_to, require_entities)
        ]
        if not episodes:
            return []

        cue_entities = {e.casefold() for e in cue.entities}
        cue_context = {t.casefold() for t in cue.topics}
        if cue.context:
            cue_context |= ranking._context_tokens(cue.context)

        results: list[RecallResult] = []
        for episode in episodes:
            semantic, bm25_rank = candidates[episode.id]
            keyword = ranking.normalize_bm25(bm25_rank) if bm25_rank >= 0 else 0.0
            factors = ranking.compute_factors(
                episode=episode,
                semantic=semantic,
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
                    score=score,
                    factors=factors,
                    reasons=ranking.explain(factors, self._config.weights),
                )
            )
        results.sort(key=lambda result: -result.score)
        return results[:k]

    @staticmethod
    def _passes_filters(
        episode,  # Episode
        source: str | None,
        time_from: datetime | None,
        time_to: datetime | None,
        require_entities: list[str] | None,
    ) -> bool:
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
