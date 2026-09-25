"""Storage adapter for explicitly asserted memory links."""

from __future__ import annotations

from datetime import datetime, timezone

from brain_memory.models import MemoryLink, NodeKind
from brain_memory.storage.db import Database


class LinkStore:
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
        row = self._db.get_link(link_id)
        return self.row_to_link(row) if row is not None else None

    def delete(self, link_id: int) -> bool:
        return self._db.delete_link(link_id)

    def for_node(self, kind: NodeKind, node_id: int) -> list[MemoryLink]:
        """All explicit links touching a node, either direction."""
        rows = [
            *self._db.links_for_source(kind.value, node_id),
            *self._db.links_for_target(kind.value, node_id),
        ]
        return [self.row_to_link(row) for row in rows]

    def row_to_link(self, row) -> MemoryLink:
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
