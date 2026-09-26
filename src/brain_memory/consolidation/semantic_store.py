"""语义记忆的存储适配器（行 <-> 模型、FTS 同步、版本链）。

每次 create/update 都会写一行 ``memory_versions``，版本链因此永远不是
可选项——"你为什么相信这个"可以端到端回答。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np

from brain_memory.models import MemoryStatus, MemoryVersion, SemanticKind, SemanticMemory
from brain_memory.storage.db import Database


def _now_iso() -> str:
    """当前 UTC ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


class SemanticStore:
    """语义记忆存储。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- 读 ----------------------------------------------------------------

    def get(self, semantic_id: int, *, with_embedding: bool = False) -> SemanticMemory | None:
        """按 id 取语义记忆；可选拿回嵌入向量。"""
        row = self._db.get_semantic(semantic_id)
        if row is None:
            return None
        return self.row_to_semantic(row, with_embedding=with_embedding)

    def get_many(self, semantic_ids: list[int]) -> list[SemanticMemory]:
        """按 id 列表批量取语义记忆（跳过缺失）。"""
        found = []
        for semantic_id in semantic_ids:
            memory = self.get(semantic_id)
            if memory is not None:
                found.append(memory)
        return found

    def get_by_concept(self, concept: str) -> SemanticMemory | None:
        """按结构键 concept 查找（大小写折叠）。"""
        row = self._db.get_semantic_by_concept(concept.casefold())
        if row is None:
            return None
        return self.row_to_semantic(row)

    def list_active(self) -> list[SemanticMemory]:
        """全部活跃语义记忆。"""
        return [self.row_to_semantic(row) for row in self._db.list_active_semantics()]

    def fts_search(self, match_expr: str, limit: int) -> list[tuple[SemanticMemory, float]]:
        """关键词候选；rank 是原始 bm25（越小越相关）。"""
        results: list[tuple[SemanticMemory, float]] = []
        for row in self._db.semantic_fts_search(match_expr, limit):
            memory = self.get(row["rowid"])
            if memory is None or memory.status is not MemoryStatus.ACTIVE:
                continue
            results.append((memory, float(row["rank"])))
        return results

    def versions(self, semantic_id: int) -> list[MemoryVersion]:
        """某语义记忆的完整版本链（按版本号升序）。"""
        return [self._row_to_version(row) for row in self._db.list_memory_versions(semantic_id)]

    # -- 写 -----------------------------------------------------------------

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
        """新建语义记忆（v1）并写首条版本行 + FTS 文档。"""
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
        metadata: dict | None = None,
    ) -> SemanticMemory:
        """更新语义记忆：版本号 +1，写新版本行，重建 FTS 文档。

        metadata 与既有 metadata 合并（新值覆盖同名键）。
        """
        now = _now_iso()
        new_version = existing.version + 1
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        merged_metadata = {**(existing.metadata or {}), **(metadata or {})}
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
            metadata=merged_metadata,
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
        # 陈述变了，FTS 文档同步重建
        self._db.semantic_fts_delete(existing.id)
        self._db.semantic_fts_insert(existing.id, statement, existing.concept)
        memory = self.get(existing.id)
        assert memory is not None
        return memory

    def touch(self, semantic_ids: list[int]) -> None:
        """再激活簿记（被召回时累加访问统计）。"""
        now = _now_iso()
        for semantic_id in set(semantic_ids):
            self._db.touch_semantic(semantic_id, now)

    def archive(self, semantic_id: int) -> bool:
        """软删除：active -> archived。"""
        return self._db.set_semantic_status(semantic_id, MemoryStatus.ARCHIVED.value)

    def forget(self, semantic_id: int) -> bool:
        """终态（仍是软的）：数据保留，restore() 仍可救回。"""
        return self._db.set_semantic_status(semantic_id, MemoryStatus.FORGOTTEN.value)

    def restore(self, semantic_id: int) -> bool:
        """恢复为 active。"""
        return self._db.set_semantic_status(semantic_id, MemoryStatus.ACTIVE.value)

    # -- 映射 ------------------------------------------------------------------

    def row_to_semantic(self, row: sqlite3.Row, *, with_embedding: bool = False) -> SemanticMemory:
        """数据库行 -> SemanticMemory 模型（可选拿回嵌入向量）。"""
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
        """数据库行 -> MemoryVersion 模型。"""
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
