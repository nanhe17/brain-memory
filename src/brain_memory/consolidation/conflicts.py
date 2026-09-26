"""冲突记录的存储适配器（行 <-> 模型）。

冲突是一等记录而非注脚：Phase 6 图谱的 ``contradicts``/``updates`` 边
将来直接从这张表生成。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from brain_memory.models import ConflictKind, ConflictStatus, MemoryConflict
from brain_memory.storage.db import Database


def _now_iso() -> str:
    """当前 UTC ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


class ConflictStore:
    """冲突记录存储。"""

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
        """新建一条 open 冲突记录。"""
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
        """按 id 取冲突记录。"""
        row = self._db.get_conflict(conflict_id)
        return self.row_to_conflict(row) if row is not None else None

    def for_semantic(self, semantic_id: int) -> list[MemoryConflict]:
        """某语义记忆的全部冲突记录。"""
        return [
            self.row_to_conflict(row)
            for row in self._db.conflicts_for_semantic(semantic_id)
        ]

    def open_semantic_ids(self) -> list[int]:
        """存在 open 冲突的语义记忆 id 列表。"""
        return self._db.open_conflict_semantic_ids()

    def list(self, status: str | None = None) -> list[MemoryConflict]:
        """全部/按状态过滤的冲突记录。"""
        rows = (
            self._db.list_conflicts_by_status(status)
            if status
            else self._db.list_conflicts()
        )
        return [self.row_to_conflict(row) for row in rows]

    def count_open(self) -> int:
        """open 冲突总数。"""
        return self._db.count_open_conflicts()

    def resolve_for_semantic(
        self,
        semantic_id: int,
        *,
        kind: ConflictKind,
        resolution_version: int,
        dismiss: bool = False,
    ) -> int:
        """结清（或驳回）某语义记忆的全部 open 冲突。"""
        return self._db.resolve_conflicts_for_semantic(
            semantic_id,
            kind=kind.value,
            resolution_version=resolution_version,
            resolved_at=_now_iso(),
            dismiss=dismiss,
        )

    def row_to_conflict(self, row: sqlite3.Row) -> MemoryConflict:
        """数据库行 -> MemoryConflict 模型。"""
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
