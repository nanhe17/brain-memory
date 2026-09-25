"""The episodic store — append-only, never merged, never overwritten.

An episode is an immutable *event*: encoding the same content twice does not
create a new memory (it bumps access statistics on the existing one), and
"updating" a memory is not an operation this store offers.  Only lifecycle
state moves: active -> archived (soft delete), archive -> active (restore).

This is the concrete realization of pattern separation at the episodic layer:
because episodes are never merged, two similar-but-different experiences stay
two rows forever.  Merge/split decisions belong to the semantic layer (Phase
3), keyed structurally, not here.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

import numpy as np

from brain_memory.models import Episode, ExtractedExperience, MemoryStatus
from brain_memory.storage.db import Database


def content_hash(content: str) -> str:
    """Stable near-duplicate key: whitespace-collapsed content, SHA-256."""
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class EpisodicStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- write ----------------------------------------------------------------

    def add(
        self,
        extracted: ExtractedExperience,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> tuple[Episode, bool]:
        """Store an experience; returns ``(episode, duplicate)``.

        ``duplicate=True`` means an active episode with the same content hash
        already existed — its access statistics were bumped instead.
        """
        digest = content_hash(extracted.content)
        existing = self._db.get_active_episode_by_hash(digest)
        now_iso = datetime.now(timezone.utc).isoformat()
        if existing is not None:
            self._db.touch_episode(existing["id"], now_iso)
            episode = self.get(int(existing["id"]))
            assert episode is not None
            return episode, True

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        episode_id = self._db.insert_episode(
            content=extracted.content,
            content_hash=digest,
            entities=extracted.entities,
            topics=extracted.topics,
            key_facts=extracted.key_facts,
            emphasis=extracted.emphasis_signals,
            context=extracted.context,
            source=extracted.source,
            created_at=extracted.timestamp.isoformat(),
            embedding=emb.tobytes(),
            embedding_dim=int(emb.shape[0]),
            importance=extracted.importance,
            confidence=extracted.confidence,
            metadata=metadata or {},
        )
        tags = [(episode_id, "entity", e.casefold()) for e in extracted.entities]
        tags += [(episode_id, "topic", t.casefold()) for t in extracted.topics]
        self._db.insert_tags(tags)
        self._db.fts_insert(
            episode_id, extracted.content,
            " ".join(extracted.entities), " ".join(extracted.topics),
        )
        episode = self.get(episode_id)
        assert episode is not None
        return episode, False

    # -- read -------------------------------------------------------------------

    def get(self, episode_id: int, *, with_embedding: bool = False) -> Episode | None:
        row = self._db.get_episode(episode_id)
        if row is None:
            return None
        return self.row_to_episode(row, with_embedding=with_embedding)

    def get_many(self, episode_ids: list[int], *, with_embedding: bool = False) -> list[Episode]:
        episodes = []
        for episode_id in episode_ids:
            episode = self.get(episode_id, with_embedding=with_embedding)
            if episode is not None:
                episodes.append(episode)
        return episodes

    def all_active(self) -> list[Episode]:
        return [self.row_to_episode(row) for row in self._db.list_active_episodes()]

    def fts_search(self, match_expr: str, limit: int) -> list[tuple[Episode, float]]:
        """Keyword candidates; rank is raw bm25 (smaller is better)."""
        results: list[tuple[Episode, float]] = []
        for row in self._db.fts_search(match_expr, limit):
            episode_row = self._db.get_episode(row["rowid"])
            if episode_row is None or episode_row["status"] != MemoryStatus.ACTIVE.value:
                continue
            results.append((self.row_to_episode(episode_row), float(row["rank"])))
        return results

    def ids_for_entities(self, entities: list[str]) -> set[int]:
        ids: set[int] = set()
        for entity in entities:
            ids.update(self._db.episode_ids_by_tag("entity", entity.casefold()))
        return ids

    def ids_for_topics(self, topics: list[str]) -> set[int]:
        ids: set[int] = set()
        for topic in topics:
            ids.update(self._db.episode_ids_by_tag("topic", topic.casefold()))
        return ids

    # -- lifecycle ---------------------------------------------------------------

    def touch(self, episode_ids: list[int]) -> None:
        """Reactivation bookkeeping for recalled memories (design doc §15)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        for episode_id in set(episode_ids):
            self._db.touch_episode(episode_id, now_iso)

    def archive(self, episode_id: int) -> bool:
        return self._db.set_episode_status(episode_id, MemoryStatus.ARCHIVED.value)

    def restore(self, episode_id: int) -> bool:
        return self._db.set_episode_status(episode_id, MemoryStatus.ACTIVE.value)

    # -- helpers -------------------------------------------------------------------

    def row_to_episode(self, row: sqlite3.Row, *, with_embedding: bool = False) -> Episode:
        embedding = None
        if with_embedding and row["embedding"] is not None:
            dim = row["embedding_dim"]
            embedding = np.frombuffer(row["embedding"], dtype=np.float32, count=dim).copy()
        return Episode(
            id=row["id"],
            content=row["content"],
            content_hash=row["content_hash"],
            entities=self._db.loads(row["entities"], []),
            topics=self._db.loads(row["topics"], []),
            key_facts=self._db.loads(row["key_facts"], []),
            emphasis_signals=self._db.loads(row["emphasis"], []),
            context=row["context"],
            source=row["source"],
            created_at=self._db.parse_dt(row["created_at"]),
            importance=row["importance"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            last_accessed=self._db.parse_dt(row["last_accessed"]),
            status=MemoryStatus(row["status"]),
            metadata=self._db.loads(row["metadata"], {}),
            embedding=embedding,
        )
