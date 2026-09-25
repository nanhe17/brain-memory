"""MemoryEngine — the facade that expresses memory *behaviors*.

Public surface (design doc §4): encode / recall / inspect / forget / restore
/ stats, plus a session-scoped working memory.  The recall pipeline is:
parse cue -> working-memory boost -> optional LLM query expansion (cue
variants retrieved in union) -> normalized factor ranking -> optional blind
LLM rerank fused into the score.  Every LLM stage is gated by config and
degrades to the deterministic path on failure; everything lifecycle-related
(timestamps, hashing, storage, stats) is pure code here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import TracebackType

import numpy as np

from brain_memory.config import MemoryConfig
from brain_memory.episodic.store import EpisodicStore
from brain_memory.embeddings.base import EmbeddingProvider
from brain_memory.embeddings.hash_embedder import HashEmbedder
from brain_memory.embeddings.openai_compatible import OpenAICompatibleEmbedder
from brain_memory.extraction.base import ExperienceParser
from brain_memory.extraction.heuristic import HeuristicExperienceParser
from brain_memory.extraction.llm import LLMExperienceParser
from brain_memory.models import (
    EncodeResult,
    EngineStats,
    Episode,
    ExtractedExperience,
    MemoryStatus,
    RecallResult,
)
from brain_memory.query.expander import LLMQueryExpander
from brain_memory.retrieval.reranker import LLMReranker
from brain_memory.retrieval.retriever import Retriever
from brain_memory.retrieval.vector_index import VectorIndex
from brain_memory.storage.db import Database
from brain_memory.working.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


def _embedding_text(content: str, entities: list[str], topics: list[str]) -> str:
    """What gets embedded: the content plus its structured tags.

    Tag augmentation noticeably improves recall for short cues (a cue like
    "minecraft mod" matches an episode that mentions Minecraft only in its
    entity list).
    """
    tags = " ".join(dict.fromkeys(entities + topics))
    return f"{content}\n{tags}" if tags else content


class MemoryEngine:
    """Owns storage, parsing, embedding, retrieval, and working memory."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig.from_env()
        self._db = Database(self.config.db_path)
        self._store = EpisodicStore(self._db)
        self._index = VectorIndex(self._db)
        self._embedder = self._build_embedder()
        self._parser = self._build_parser()
        self._retriever = Retriever(self._store, self._index, self.config)
        self._expander = self._build_expander()
        self._reranker = self._build_reranker()
        self.working = WorkingMemory()

    # -- construction ---------------------------------------------------------

    def _build_embedder(self) -> EmbeddingProvider:
        if self.config.embedding_provider == "openai_compatible":
            return OpenAICompatibleEmbedder(
                base_url=self.config.api_base,
                api_key=self.config.api_key,
                model=self.config.embedding_model,
            )
        return HashEmbedder(dim=self.config.embedding_dim)

    def _build_parser(self) -> ExperienceParser:
        heuristic = HeuristicExperienceParser()
        if self.config.llm_model and self.config.llm_api_key:
            return LLMExperienceParser(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                fallback=heuristic,
            )
        return heuristic

    def _llm_ready(self) -> bool:
        return bool(self.config.llm_model and self.config.llm_api_key)

    def _build_expander(self) -> LLMQueryExpander | None:
        if self.config.query_expansion == "auto" and self._llm_ready():
            return LLMQueryExpander(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
            )
        return None

    def _build_reranker(self) -> LLMReranker | None:
        if self.config.rerank == "auto" and self._llm_ready():
            return LLMReranker(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                timeout=self.config.rerank_timeout,
            )
        return None

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder

    # -- core behaviors ---------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        source: str = "conversation",
        context: str | None = None,
        created_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> EncodeResult:
        """Parse, embed, and store an experience (append-only; see store)."""
        extracted = self._parser.parse(text, source=source, context=context, timestamp=created_at)
        vector = self._embed(extracted)
        episode, duplicate = self._store.add(extracted, vector, metadata=metadata)
        if not duplicate:
            self._index.invalidate()
        self.working.note_episode(episode.id)
        self.working.add_entities(episode.entities)
        return EncodeResult(episode=episode, duplicate=duplicate)

    def recall(
        self,
        cue: str,
        *,
        k: int | None = None,
        source: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        require_entities: list[str] | None = None,
        touch: bool = True,
        use_working_memory: bool = True,
    ) -> list[RecallResult]:
        """Retrieve memories for a (possibly partial) cue, with explanations.

        ``touch=True`` reactivation-bookkeeps the hits: access counts rise,
        which feeds the frequency factor on future recalls (doc §15).
        """
        final_k = k or self.config.default_top_k
        parsed_cue = self._parser.parse(cue)
        boost = self._working_memory_entities() if use_working_memory else None

        variants, expanded_from, expanded_to = self._expand_query(parsed_cue, time_from, time_to)
        vectors = self._embed_variants(variants)
        limit = self.config.rerank_top_n if self._reranker is not None else final_k
        results = self._retriever.retrieve(
            variants,
            vectors,
            k=limit,
            source=source,
            time_from=expanded_from or time_from,
            time_to=expanded_to or time_to,
            require_entities=require_entities,
            entity_boost=boost,
        )
        if self._reranker is not None and len(results) >= 2:
            results = self._fuse_rerank(cue, results)
        results = results[:final_k]
        if touch and results:
            self._store.touch([result.episode.id for result in results])
        self.working.remember_recall(results)
        return results

    # -- recall pipeline stages -------------------------------------------------

    def _working_memory_entities(self) -> list[str] | None:
        """Session entities that bias the entity-overlap factor toward
        memories related to the current task.  Factor-only by design: they
        must never widen the query channels (a stale session entity in the
        FTS terms would resurrect unrelated memories)."""
        entities = self.working.state.active_entities[:5]
        return entities or None

    def _expand_query(
        self,
        parsed_cue: ExtractedExperience,
        time_from: datetime | None,
        time_to: datetime | None,
    ) -> tuple[list[ExtractedExperience], datetime | None, datetime | None]:
        """Original cue plus optional LLM expansion variants (advisory only:
        the original is always kept as its own retrieval channel)."""
        variants = [parsed_cue]
        expanded_from = expanded_to = None
        if self._expander is not None:
            try:
                expansion = self._expander.expand(parsed_cue.content, self.working.snapshot())
            except Exception as exc:  # noqa: BLE001 — expansion is advisory
                logger.warning("query expansion failed (%s); using original cue", exc)
                expansion = None
            if expansion is not None:
                expanded_from = self._parse_iso(expansion.time_from)
                expanded_to = self._parse_iso(expansion.time_to)
                for text in [expansion.rewritten, *expansion.sub_queries[:2]]:
                    text = (text or "").strip()
                    if not text or any(text == variant.content for variant in variants):
                        continue
                    try:
                        variants.append(self._parser.parse(text))
                    except ValueError:
                        continue
        return variants, expanded_from, expanded_to

    def _embed_variants(self, variants: list[ExtractedExperience]) -> list[np.ndarray]:
        texts = [_embedding_text(v.content, v.entities, v.topics) for v in variants]
        return list(self._embedder.embed_texts(texts))

    def _fuse_rerank(self, cue: str, results: list[RecallResult]) -> list[RecallResult]:
        """Blind LLM relevance fused into the factor score (reranker sees
        content only).  Any failure keeps the pure factor ranking."""
        try:
            scores = self._reranker.rerank(cue, results)
        except Exception as exc:  # noqa: BLE001 — reranking is advisory
            logger.warning("rerank failed (%s); keeping factor ranking", exc)
            return results
        mix = self.config.rerank_mix
        fused: list[RecallResult] = []
        for result, relevance in zip(results, scores):
            new_score = mix * relevance + (1.0 - mix) * result.score
            fused.append(
                result.model_copy(
                    update={
                        "llm_relevance": relevance,
                        "score": new_score,
                        "reasons": [
                            *result.reasons,
                            f"llm_relevance: {relevance:.2f} × w{mix:.2f} → {mix * relevance:.2f}",
                        ],
                    }
                )
            )
        fused.sort(key=lambda result: -result.score)
        return fused

    @staticmethod
    def _parse_iso(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def inspect(self, memory_id: int) -> dict | None:
        """Full memory details plus the most related active memories."""
        episode = self._store.get(memory_id, with_embedding=True)
        if episode is None:
            return None
        related: list[Episode] = []
        if episode.embedding is not None:
            for related_id, _similarity in self._index.search(episode.embedding, 4):
                if related_id != episode.id:
                    related_episode = self._store.get(related_id)
                    if related_episode is not None:
                        related.append(related_episode)
                if len(related) >= 3:
                    break
        return {
            "episode": episode,
            "related": related,
        }

    def forget(self, memory_id: int) -> bool:
        """Soft delete: archive now, physical removal is a later-phase decision."""
        archived = self._store.archive(memory_id)
        if archived:
            self._index.invalidate()
        return archived

    def restore(self, memory_id: int) -> bool:
        restored = self._store.restore(memory_id)
        if restored:
            self._index.invalidate()
        return restored

    def stats(self) -> EngineStats:
        counts = self._db.count_by_status()
        extremes = self._db.episode_extremes()
        return EngineStats(
            total_episodes=sum(counts.values()),
            active=counts.get(MemoryStatus.ACTIVE.value, 0),
            archived=counts.get(MemoryStatus.ARCHIVED.value, 0),
            forgotten=counts.get(MemoryStatus.FORGOTTEN.value, 0),
            distinct_entities=self._distinct_entity_count(),
            avg_importance=round(float(extremes["avg_importance"] or 0.0), 4),
            oldest_created_at=self._db.parse_dt(extremes["oldest"]),
            newest_created_at=self._db.parse_dt(extremes["newest"]),
        )

    # -- helpers ------------------------------------------------------------------

    def _distinct_entity_count(self) -> int:
        rows = self._db.list_active_episodes()
        seen: set[str] = set()
        for row in rows:
            for entity in self._db.loads(row["entities"], []):
                seen.add(entity.casefold())
        return len(seen)

    def _embed(self, extracted: ExtractedExperience) -> np.ndarray:
        text = _embedding_text(extracted.content, extracted.entities, extracted.topics)
        return self._embedder.embed_texts([text])[0]

    # -- lifecycle ------------------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "MemoryEngine":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
