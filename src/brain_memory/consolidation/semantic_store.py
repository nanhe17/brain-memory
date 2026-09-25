"""Storage adapter for semantic memories (rows <-> models, FTS sync, versions).

Every create/update writes a ``memory_versions`` row, so the version trail is
never optional — "why do you believe this" is answerable end to end.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np

from brain_memory.models import MemoryStatus, MemoryVersion, SemanticKind, SemanticMemory
from brain_memory.storage.db import Database


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SemanticStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- read ----------------------------------------------------------------

    def get(self, semantic_id: int, *, with_embedding: bool = False) -> SemanticMemory | None:
        row = self._db.get_semantic(semantic_id)
        if row is None:
            return None
        return self.row_to_semantic(row, with_embedding=with_embedding)

    def get_many(self, semantic_ids: list[int]) -> list[SemanticMemory]:
        found = []
        for semantic_id in semantic_ids:
            memory = self.get(semantic_id)
            if memory is not None:
                found.append(memory)
        return found

    def get_by_concept(self, concept: str) -> SemanticMemory | None:
        row = self._db.get_semantic_by_concept(concept.casefold())
        if row is None:
            return None
        return self.row_to_semantic(row)

    def list_active(self) -> list[SemanticMemory]:
        return [self.row_to_semantic(row) for row in self._db.list_active_semantics()]

    def fts_search(self, match_expr: str, limit: int) -> list[tuple[SemanticMemory, float]]:
        """Keyword candidates; rank is raw bm25 (smaller is better)."""
        results: list[tuple[SemanticMemory, float]] = []
        for row in self._db.semantic_fts_search(match_expr, limit):
            memory = self.get(row["rowid"])
            if memory is None or memory.status is not MemoryStatus.ACTIVE:
                continue
            results.append((memory, float(row["rank"])))
        return results

    def versions(self, semantic_id: int) -> list[MemoryVersion]:
        return [self._row_to_version(row) for row in self._db.list_memory_versions(semantic_id)]

    # -- write -----------------------------------------------------------------

    def create(
        self,
        *,
        concept: str,
        kind: SemanticKind,
        statement: str,
        confidence: float,
        evidence_ids: list[int],
        embedding: np.ndarray,
        metadata: dict | None = None,
        change_reason: str = "initial consolidation",
    ) -> SemanticMemory:
        now = _now_iso()
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        semantic_id = self._db.insert_semantic(
            concept=concept.casefold(),
            kind=kind.value,
            statement=statement,
            confidence=confidence,
            evidence_ids=evidence_ids,
            created_at=now,
            embedding=vector.tobytes(),
            embedding_dim=int(vector.shape[0]),
            metadata=metadata or {},
        )
        self._db.insert_memory_version(
            semantic_id=semantic_id,
            version=1,
            statement=statement,
            confidence=confidence,
            evidence_ids=evidence_ids,
            created_at=now,
            change_reason=change_reason,
        )
        self._db.semantic_fts_insert(semantic_id, statement, concept.casefold())
        memory = self.get(semantic_id)
        assert memory is not None
        return memory

    def update(
        self,
        existing: SemanticMemory,
        *,
        kind: SemanticKind,
        statement: str,
        confidence: float,
        evidence_ids: list[int],
        embedding: np.ndarray,
        change_reason: str,
    ) -> SemanticMemory:
        now = _now_iso()
        new_version = existing.version + 1
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self._db.update_semantic(
            existing.id,
            kind=kind.value,
            statement=statement,
            confidence=confidence,
            evidence_ids=evidence_ids,
            updated_at=now,
            version=new_version,
            embedding=vector.tobytes(),
            embedding_dim=int(vector.shape[0]),
        )
        self._db.insert_memory_version(
            semantic_id=existing.id,
            version=new_version,
            statement=statement,
            confidence=confidence,
            evidence_ids=evidence_ids,
            created_at=now,
            change_reason=change_reason,
        )
        self._db.semantic_fts_delete(existing.id)
        self._db.semantic_fts_insert(existing.id, statement, existing.concept)
        memory = self.get(existing.id)
        assert memory is not None
        return memory

    def touch(self, semantic_ids: list[int]) -> None:
        now = _now_iso()
        for semantic_id in set(semantic_ids):
            self._db.touch_semantic(semantic_id, now)

    def archive(self, semantic_id: int) -> bool:
        return self._db.set_semantic_status(semantic_id, MemoryStatus.ARCHIVED.value)

    def restore(self, semantic_id: int) -> bool:
        return self._db.set_semantic_status(semantic_id, MemoryStatus.ACTIVE.value)

    # -- mapping ------------------------------------------------------------------

    def row_to_semantic(self, row: sqlite3.Row, *, with_embedding: bool = False) -> SemanticMemory:
        embedding = None
        if with_embedding and row["embedding"] is not None:
            dim = row["embedding_dim"]
            embedding = np.frombuffer(row["embedding"], dtype=np.float32, count=dim).copy()
        return SemanticMemory(
            id=row["id"],
            concept=row["concept"],
            kind=SemanticKind(row["kind"]),
            statement=row["statement"],
            confidence=row["confidence"],
            evidence_ids=self._db.loads(row["evidence_ids"], []),
            created_at=self._db.parse_dt(row["created_at"]),
            updated_at=self._db.parse_dt(row["updated_at"]),
            version=row["version"],
            status=MemoryStatus(row["status"]),
            access_count=row["access_count"],
            last_accessed=self._db.parse_dt(row["last_accessed"]),
            metadata=self._db.loads(row["metadata"], {}),
            embedding=embedding,
        )

    def _row_to_version(self, row: sqlite3.Row) -> MemoryVersion:
        return MemoryVersion(
            id=row["id"],
            semantic_id=row["semantic_id"],
            version=row["version"],
            statement=row["statement"],
            confidence=row["confidence"],
            evidence_ids=self._db.loads(row["evidence_ids"], []),
            created_at=self._db.parse_dt(row["created_at"]),
            change_reason=row["change_reason"],
        )
