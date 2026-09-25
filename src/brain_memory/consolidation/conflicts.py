"""Storage adapter for memory conflicts (rows <-> models).

Conflicts are first-class records, not annotations: Phase 6's Memory Graph
``contradicts``/``updates`` edges will be built directly from this table.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from brain_memory.models import ConflictKind, ConflictStatus, MemoryConflict
from brain_memory.storage.db import Database


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConflictStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        *,
        semantic_id: int,
        kind: ConflictKind,
        old_version: int,
        statement_before: str | None,
        trigger_episode_id: int | None,
        trigger_kind: str,
        metadata: dict | None = None,
    ) -> MemoryConflict:
        conflict_id = self._db.insert_conflict(
            semantic_id=semantic_id,
            kind=kind.value,
            status=ConflictStatus.OPEN.value,
            old_version=old_version,
            statement_before=statement_before,
            trigger_episode_id=trigger_episode_id,
            trigger_kind=trigger_kind,
            detected_at=_now_iso(),
            metadata=metadata or {},
        )
        conflict = self.get(conflict_id)
        assert conflict is not None
        return conflict

    def get(self, conflict_id: int) -> MemoryConflict | None:
        row = self._db.get_conflict(conflict_id)
        if row is None:
            return None
        return self.row_to_conflict(row)

    def for_semantic(self, semantic_id: int) -> list[MemoryConflict]:
        return [
            self.row_to_conflict(row)
            for row in self._db.conflicts_for_semantic(semantic_id)
        ]

    def open_semantic_ids(self) -> list[int]:
        return self._db.open_conflict_semantic_ids()

    def list(self, status: str | None = None) -> list[MemoryConflict]:
        rows = (
            self._db.list_conflicts_by_status(status)
            if status
            else self._db.list_conflicts()
        )
        return [self.row_to_conflict(row) for row in rows]

    def count_open(self) -> int:
        return self._db.count_open_conflicts()

    def resolve_for_semantic(
        self,
        semantic_id: int,
        *,
        kind: ConflictKind,
        resolution_version: int,
        dismiss: bool = False,
    ) -> int:
        """Resolve (or dismiss) all open conflicts of one semantic memory."""
        return self._db.resolve_conflicts_for_semantic(
            semantic_id,
            kind=kind.value,
            resolution_version=resolution_version,
            resolved_at=_now_iso(),
            dismiss=dismiss,
        )

    def row_to_conflict(self, row: sqlite3.Row) -> MemoryConflict:
        return MemoryConflict(
            id=row["id"],
            semantic_id=row["semantic_id"],
            kind=ConflictKind(row["kind"]),
            status=ConflictStatus(row["status"]),
            old_version=row["old_version"],
            statement_before=row["statement_before"],
            trigger_episode_id=row["trigger_episode_id"],
            trigger_kind=row["trigger_kind"],
            detected_at=self._db.parse_dt(row["detected_at"]),
            resolution_version=row["resolution_version"],
            resolved_at=self._db.parse_dt(row["resolved_at"]),
            metadata=self._db.loads(row["metadata"], {}),
        )
