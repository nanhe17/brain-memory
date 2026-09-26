"""情景存储——append-only，永不合并，永不被覆写。

episode 是不可变的*事件*：相同内容编码两次不会新建记忆（而是给既有
记忆累加访问统计——同一事件被再次目击），"更新记忆"在这个存储里不是
一个操作。只有生命周期状态会移动：active → archived（软删除）、
archive → active（恢复）。

这是模式分离在情景层的具体实现：因为 episode 永不合并，两条相似但
不同的经历永远是两行。"合并还是分立"的决策属于语义层（Phase 3）——
那里用结构化键去重——而不属于这里。
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

import numpy as np

from brain_memory.models import Episode, ExtractedExperience, MemoryStatus
from brain_memory.storage.db import Database


def content_hash(content: str) -> str:
    """稳定的近似去重键：空白归一化后的内容取 SHA-256。"""
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class EpisodicStore:
    """情景记忆的存储适配器（行 <-> 模型、FTS 同步、生命周期）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- 写 ----------------------------------------------------------------

    def add(
        self,
        extracted: ExtractedExperience,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> tuple[Episode, bool]:
        """存储一条经验；返回 ``(episode, duplicate)``。

        ``duplicate=True`` 表示已有相同内容哈希的活跃 episode——此时
        只累加其访问统计，不新建行。
        """
        digest = content_hash(extracted.content)
        # 近重复检测：仅精确重复抑制，绝不模糊合并
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
        # 标签写入统一 casefold，保证实体索引的大小写一致查询
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

    # -- 读 -------------------------------------------------------------------

    def get(self, episode_id: int, *, with_embedding: bool = False) -> Episode | None:
        """按 id 取 episode；可选拿回嵌入向量。"""
        row = self._db.get_episode(episode_id)
        if row is None:
            return None
        return self.row_to_episode(row, with_embedding=with_embedding)

    def get_many(self, episode_ids: list[int], *, with_embedding: bool = False) -> list[Episode]:
        """按 id 列表批量取 episode（保持顺序，跳过缺失）。"""
        episodes = []
        for episode_id in episode_ids:
            episode = self.get(episode_id, with_embedding=with_embedding)
            if episode is not None:
                episodes.append(episode)
        return episodes

    def all_active(self) -> list[Episode]:
        """全部活跃 episode。"""
        return [self.row_to_episode(row) for row in self._db.list_active_episodes()]

    def fts_search(self, match_expr: str, limit: int) -> list[tuple[Episode, float]]:
        """关键词候选；rank 是原始 bm25（越小越相关）。"""
        results: list[tuple[Episode, float]] = []
        for row in self._db.fts_search(match_expr, limit):
            episode_row = self._db.get_episode(row["rowid"])
            # 非 active 的行直接跳过（归档/遗忘不参与检索）
            if episode_row is None or episode_row["status"] != MemoryStatus.ACTIVE.value:
                continue
            results.append((self.row_to_episode(episode_row), float(row["rank"])))
        return results

    def ids_for_entities(self, entities: list[str]) -> set[int]:
        """按实体标签反查 episode id 集合。"""
        ids: set[int] = set()
        for entity in entities:
            ids.update(self._db.episode_ids_by_tag("entity", entity.casefold()))
        return ids

    def ids_for_topics(self, topics: list[str]) -> set[int]:
        """按话题标签反查 episode id 集合。"""
        ids: set[int] = set()
        for topic in topics:
            ids.update(self._db.episode_ids_by_tag("topic", topic.casefold()))
        return ids

    # -- 生命周期 ---------------------------------------------------------------

    def touch(self, episode_ids: list[int]) -> None:
        """被召回记忆的再激活簿记（设计文档 §15）。"""
        now_iso = datetime.now(timezone.utc).isoformat()
        for episode_id in set(episode_ids):
            self._db.touch_episode(episode_id, now_iso)

    def archive(self, episode_id: int) -> bool:
        """软删除：active -> archived。"""
        return self._db.set_episode_status(episode_id, MemoryStatus.ARCHIVED.value)

    def forget(self, episode_id: int) -> bool:
        """终态（仍是软的）：数据保留，restore() 仍可救回。"""
        return self._db.set_episode_status(episode_id, MemoryStatus.FORGOTTEN.value)

    def restore(self, episode_id: int) -> bool:
        """恢复为 active。"""
        return self._db.set_episode_status(episode_id, MemoryStatus.ACTIVE.value)

    # -- 辅助 -------------------------------------------------------------------

    def row_to_episode(self, row: sqlite3.Row, *, with_embedding: bool = False) -> Episode:
        """数据库行 -> Episode 模型（可选拿回嵌入向量）。"""
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
