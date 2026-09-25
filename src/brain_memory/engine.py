"""MemoryEngine — the facade that expresses memory *behaviors*.

Public surface (design doc §4): encode / recall / inspect / forget / restore
/ stats, plus a session-scoped working memory.  Everything lifecycle-related
(timestamps, hashing, storage, stats) is deterministic code here; LLMs only
ever assist inside the parser.
"""

from __future__ import annotations

from datetime import datetime
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
from brain_memory.retrieval.retriever import Retriever
from brain_memory.retrieval.vector_index import VectorIndex
from brain_memory.storage.db import Database
from brain_memory.working.working_memory import WorkingMemory


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
    ) -> list[RecallResult]:
        """Retrieve memories for a (possibly partial) cue, with explanations.

        ``touch=True`` reactivation-bookkeeps the hits: access counts rise,
        which feeds the frequency factor on future recalls (doc §15).
        """
        parsed_cue = self._parser.parse(cue)
        vector = self._embed(parsed_cue)
        results = self._retriever.retrieve(
            parsed_cue,
            vector,
            k,
            source=source,
            time_from=time_from,
            time_to=time_to,
            require_entities=require_entities,
        )
        if touch and results:
            self._store.touch([result.episode.id for result in results])
        self.working.remember_recall(results)
        return results

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
