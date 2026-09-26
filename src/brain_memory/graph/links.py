"""显式断言关系（memory_links）的存储适配器。"""

from __future__ import annotations

from datetime import datetime, timezone

from brain_memory.models import MemoryLink, NodeKind
from brain_memory.storage.db import Database


def _now_iso() -> str:
    """当前 UTC ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


class LinkStore:
    """显式关系边的存储。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        *,
        source_kind: NodeKind,
        source_id: int,
        target_kind: NodeKind,
        target_id: int,
        relation: str,
        weight: float,
        created_by: str,
        metadata: dict | None = None,
    ) -> MemoryLink:
        """新建一条显式边。"""
        link_id = self._db.insert_link(
            source_kind=source_kind.value,
            source_id=source_id,
            target_kind=target_kind.value,
            target_id=target_id,
            relation=relation,
            weight=weight,
            created_by=created_by,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        link = self.get(link_id)
        assert link is not None
        return link

    def get(self, link_id: int) -> MemoryLink | None:
        """按 id 取边。"""
        row = self._db.get_link(link_id)
        return self.row_to_link(row) if row is not None else None

    def delete(self, link_id: int) -> bool:
        """删除一条边。"""
        return self._db.delete_link(link_id)

    def for_node(self, kind: NodeKind, node_id: int) -> list[MemoryLink]:
        """触及某节点的全部显式边（双向）。"""
        rows = [
            *self._db.links_for_source(kind.value, node_id),
            *self._db.links_for_target(kind.value, node_id),
        ]
        return [self.row_to_link(row) for row in rows]

    def row_to_link(self, row) -> MemoryLink:
        """数据库行 -> MemoryLink 模型。"""
        return MemoryLink(
            id=row["id"],
            source_kind=NodeKind(row["source_kind"]),
            source_id=row["source_id"],
            target_kind=NodeKind(row["target_kind"]),
            target_id=row["target_id"],
            relation=row["relation"],
            weight=row["weight"],
            created_by=row["created_by"],
            created_at=self._db.parse_dt(row["created_at"]),
            metadata=self._db.loads(row["metadata"], {}),
        )
