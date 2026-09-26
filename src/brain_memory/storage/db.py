"""SQLite 存储层（仓储风格）。

设计要点
--------
* 每条 SQL 都是调用点内联的单行字面量，仅使用 ``?`` 占位符——不使用任何
  字符串格式化、拼接或变量 SQL，调用方无法改变查询结构。schema 迁移也
  逐条写出的原因相同，且每列在插入时都显式赋值（不需要 DEFAULT 子句）。
* 写路径串行化在一把锁之后（Phase 1 即明确的单写者假设；PostgreSQL 之后
  可以在同一接口下替换）。
* FTS5 提供关键词检索，零额外依赖。情景 FTS 表只在写入时手动同步
  （insert）；归档/遗忘不删 FTS 行，而是在读取时按状态过滤（见
  EpisodicStore.fts_search）——所以 restore() 能原样带回可检索行。
* ``episode_tags`` 是实体/话题索引，用于元数据过滤与后续巩固分组。
* ``semantic_memories`` 现在就建表（Phase 3 写入），避免巩固落地时
  schema 来回变更。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utcnow_iso() -> str:
    """当前 UTC 时间的 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """带顺序 schema 迁移的线程安全 sqlite3 封装。

    数据访问方法都是命名且窄接口的；SQL 内联在每个调用点，一律使用
    绑定参数。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：所有访问都经过 self._lock。
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # -- 迁移 ---------------------------------------------------------

    def _migrate(self) -> None:
        """按序应用 schema 迁移。

        迁移 1 创建基础 schema；迁移 2 以 Phase 3 的完整形态重建
        ``semantic_memories``（该表在巩固出现前从未被写入，drop+create
        安全）并新建巩固簿记表；迁移 3 新增冲突记录表；迁移 4 新增
        显式关系表。
        """
        with self._lock:
            # 先建版本表再查版本（避免自引用）
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
            self._conn.commit()
            row = self._conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM schema_version").fetchone()
            current = row["v"]
            if current < 1:
                self._apply_migration_1()
                self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (1,))
                self._conn.commit()
            if current < 2:
                self._apply_migration_2()
                self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (2,))
                self._conn.commit()
            if current < 3:
                self._apply_migration_3()
                self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (3,))
                self._conn.commit()
            if current < 4:
                self._apply_migration_4()
                self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (4,))
                self._conn.commit()

    def _apply_migration_1(self) -> None:
        """迁移 1：基础表（episodes + FTS + tags + 语义表占位）。"""
        self._conn.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT NOT NULL, entities TEXT NOT NULL, topics TEXT NOT NULL, key_facts TEXT NOT NULL, emphasis TEXT NOT NULL, context TEXT, source TEXT NOT NULL, created_at TEXT NOT NULL, embedding BLOB, embedding_dim INTEGER NOT NULL, importance REAL NOT NULL, confidence REAL NOT NULL, access_count INTEGER NOT NULL, last_accessed TEXT, status TEXT NOT NULL, metadata TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(created_at)")
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(content, entities_text, topics_text)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS episode_tags (episode_id INTEGER NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_value ON episode_tags(kind, value)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS semantic_memories (id INTEGER PRIMARY KEY, concept TEXT NOT NULL, statement TEXT NOT NULL, confidence REAL NOT NULL, evidence_ids TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL, metadata TEXT NOT NULL)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")

    def _apply_migration_2(self) -> None:
        """迁移 2：重建语义表（补 kind/嵌入/访问统计列）+ 巩固簿记表。"""
        self._conn.execute("DROP TABLE IF EXISTS semantic_memories")
        self._conn.execute("CREATE TABLE IF NOT EXISTS semantic_memories (id INTEGER PRIMARY KEY, concept TEXT NOT NULL, kind TEXT NOT NULL, statement TEXT NOT NULL, confidence REAL NOT NULL, evidence_ids TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL, embedding BLOB, embedding_dim INTEGER NOT NULL, access_count INTEGER NOT NULL, last_accessed TEXT, metadata TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_concept ON semantic_memories(concept)")
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(statement, concept)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS consolidation_state (value TEXT PRIMARY KEY, representative_kind TEXT NOT NULL, last_consolidated_at TEXT NOT NULL, episode_count INTEGER NOT NULL, semantic_id INTEGER)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS memory_versions (id INTEGER PRIMARY KEY, semantic_id INTEGER NOT NULL, version INTEGER NOT NULL, statement TEXT NOT NULL, confidence REAL NOT NULL, evidence_ids TEXT NOT NULL, created_at TEXT NOT NULL, change_reason TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_semantic ON memory_versions(semantic_id, version)")

    def _apply_migration_3(self) -> None:
        """迁移 3：冲突记录表（Phase 4 再巩固的审计层）。"""
        self._conn.execute("CREATE TABLE IF NOT EXISTS memory_conflicts (id INTEGER PRIMARY KEY, semantic_id INTEGER NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, old_version INTEGER NOT NULL, statement_before TEXT, trigger_episode_id INTEGER, trigger_kind TEXT NOT NULL, detected_at TEXT NOT NULL, resolution_version INTEGER, resolved_at TEXT, metadata TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_semantic ON memory_conflicts(semantic_id, status)")

    def _apply_migration_4(self) -> None:
        """迁移 4：显式关系表（Phase 6 图谱的 agent 断言边）。"""
        self._conn.execute("CREATE TABLE IF NOT EXISTS memory_links (id INTEGER PRIMARY KEY, source_kind TEXT NOT NULL, source_id INTEGER NOT NULL, target_kind TEXT NOT NULL, target_id INTEGER NOT NULL, relation TEXT NOT NULL, weight REAL NOT NULL, created_by TEXT NOT NULL, created_at TEXT NOT NULL, metadata TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_kind, source_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_kind, target_id)")

    # -- json / datetime 辅助 --------------------------------------------

    @staticmethod
    def dumps(value: Any) -> str:
        """序列化为 JSON（保留非 ASCII 字符）。"""
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def loads(text: str | None, default: Any) -> Any:
        """反序列化 JSON，空值/损坏数据回退到默认值。"""
        if not text:
            return default
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def parse_dt(text: str | None) -> datetime | None:
        """ISO 字符串 -> datetime；naive 时间补 UTC 时区。"""
        if not text:
            return None
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    # -- episodes -------------------------------------------------------------

    def insert_episode(self, *, content: str, content_hash: str, entities: list[str],
                       topics: list[str], key_facts: list[str], emphasis: list[str],
                       context: str | None, source: str, created_at: str,
                       embedding: bytes | None, embedding_dim: int,
                       importance: float, confidence: float,
                       metadata: dict[str, Any]) -> int:
        """插入一条新 episode（append-only），返回自增 id。"""
        with self._lock:
            cur = self._conn.execute("INSERT INTO episodes (content, content_hash, entities, topics, key_facts, emphasis, context, source, created_at, embedding, embedding_dim, importance, confidence, access_count, last_accessed, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 'active', ?)", (content, content_hash, self.dumps(entities), self.dumps(topics), self.dumps(key_facts), self.dumps(emphasis), context, source, created_at, embedding, embedding_dim, importance, confidence, self.dumps(metadata)))
            self._conn.commit()
            return int(cur.lastrowid)

    def get_episode(self, episode_id: int) -> sqlite3.Row | None:
        """按 id 取单个 episode 行。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()

    def get_active_episode_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        """按内容哈希查活跃 episode（编码去重用）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes WHERE content_hash = ? AND status = 'active'", (content_hash,)).fetchone()

    def touch_episode(self, episode_id: int, when_iso: str) -> None:
        """再激活簿记：访问统计 +1（设计文档 §15）。"""
        with self._lock:
            self._conn.execute("UPDATE episodes SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (when_iso, episode_id))
            self._conn.commit()

    def set_episode_status(self, episode_id: int, status: str) -> bool:
        """迁移生命周期状态（软删除/遗忘/恢复共用）。"""
        with self._lock:
            cur = self._conn.execute("UPDATE episodes SET status = ? WHERE id = ?", (status, episode_id))
            self._conn.commit()
            return cur.rowcount > 0

    def list_active_episodes(self) -> list[sqlite3.Row]:
        """全部活跃 episode（时间倒序）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes WHERE status = 'active' ORDER BY created_at DESC").fetchall()

    def list_archived_episodes(self) -> list[sqlite3.Row]:
        """全部已归档 episode（遗忘驻留检查的数据源）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes WHERE status = 'archived' ORDER BY created_at DESC").fetchall()

    def list_all_episodes(self) -> list[sqlite3.Row]:
        """全部 episode（任意状态，时间倒序）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes ORDER BY created_at DESC").fetchall()

    def list_active_embeddings(self) -> list[sqlite3.Row]:
        """活跃 episode 的 (id, 向量, 维度)——向量索引的数据源。"""
        with self._lock:
            return self._conn.execute("SELECT id, embedding, embedding_dim FROM episodes WHERE status = 'active' AND embedding IS NOT NULL").fetchall()

    def count_by_status(self) -> dict[str, int]:
        """按生命周期状态分组计数。"""
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM episodes GROUP BY status").fetchall()
        return {row["status"]: row["n"] for row in rows}

    def episode_extremes(self) -> sqlite3.Row:
        """最早/最晚创建时间与平均重要性（统计用）。"""
        with self._lock:
            return self._conn.execute("SELECT MIN(created_at) AS oldest, MAX(created_at) AS newest, AVG(importance) AS avg_importance FROM episodes").fetchone()

    # -- tags（实体 / 话题索引） ------------------------------------------

    def insert_tags(self, tags: list[tuple[int, str, str]]) -> None:
        """批量写入 (episode_id, kind, value) 标签。"""
        if not tags:
            return
        with self._lock:
            self._conn.executemany("INSERT INTO episode_tags(episode_id, kind, value) VALUES (?, ?, ?)", tags)
            self._conn.commit()

    def delete_tags(self, episode_id: int) -> None:
        """删除某 episode 的全部标签。"""
        with self._lock:
            self._conn.execute("DELETE FROM episode_tags WHERE episode_id = ?", (episode_id,))
            self._conn.commit()

    def episode_ids_by_tag(self, kind: str, value: str) -> list[int]:
        """按标签反查 episode id 集合。"""
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT episode_id FROM episode_tags WHERE kind = ? AND value = ?", (kind, value)).fetchall()
        return [row["episode_id"] for row in rows]

    # -- 全文检索 -------------------------------------------------------

    def fts_insert(self, rowid: int, content: str, entities_text: str, topics_text: str) -> None:
        """向情景 FTS 表写入一行文档。"""
        with self._lock:
            self._conn.execute("INSERT INTO episodes_fts(rowid, content, entities_text, topics_text) VALUES (?, ?, ?, ?)", (rowid, content, entities_text, topics_text))
            self._conn.commit()

    def fts_delete(self, rowid: int) -> None:
        """从情景 FTS 表删除一行。"""
        with self._lock:
            self._conn.execute("DELETE FROM episodes_fts WHERE rowid = ?", (rowid,))
            self._conn.commit()

    def fts_search(self, match_expr: str, limit: int) -> list[sqlite3.Row]:
        """全文检索：返回 (rowid, rank)；bm25 rank 越小越相关。"""
        with self._lock:
            return self._conn.execute("SELECT rowid, bm25(episodes_fts) AS rank FROM episodes_fts WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?", (match_expr, limit)).fetchall()

    # -- semantic memories -----------------------------------------------------

    def get_semantic(self, semantic_id: int) -> sqlite3.Row | None:
        """按 id 取语义记忆行。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (semantic_id,)).fetchone()

    def get_semantic_by_concept(self, concept: str) -> sqlite3.Row | None:
        """按结构键 concept 取语义记忆（每个概念至多一行）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM semantic_memories WHERE concept = ?", (concept,)).fetchone()

    def list_active_semantics(self) -> list[sqlite3.Row]:
        """全部活跃语义记忆（更新时间倒序）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM semantic_memories WHERE status = 'active' ORDER BY updated_at DESC").fetchall()

    def list_archived_semantics(self) -> list[sqlite3.Row]:
        """全部已归档语义记忆（遗忘驻留检查用）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM semantic_memories WHERE status = 'archived' ORDER BY updated_at DESC").fetchall()

    def list_active_semantic_embeddings(self) -> list[sqlite3.Row]:
        """活跃语义记忆的向量（语义向量索引的数据源）。"""
        with self._lock:
            return self._conn.execute("SELECT id, embedding, embedding_dim FROM semantic_memories WHERE status = 'active' AND embedding IS NOT NULL").fetchall()

    def insert_semantic(self, *, concept: str, kind: str, statement: str, confidence: float,
                        evidence_ids: list[int], created_at: str, embedding: bytes | None,
                        embedding_dim: int, metadata: dict[str, Any]) -> int:
        """新建语义记忆（version=1, active），返回自增 id。"""
        with self._lock:
            cur = self._conn.execute("INSERT INTO semantic_memories (concept, kind, statement, confidence, evidence_ids, created_at, updated_at, version, status, embedding, embedding_dim, access_count, last_accessed, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'active', ?, ?, 0, NULL, ?)", (concept, kind, statement, confidence, self.dumps(evidence_ids), created_at, created_at, embedding, embedding_dim, self.dumps(metadata)))
            self._conn.commit()
            return int(cur.lastrowid)

    def update_semantic(self, semantic_id: int, *, kind: str, statement: str, confidence: float,
                        evidence_ids: list[int], updated_at: str, version: int,
                        embedding: bytes | None, embedding_dim: int,
                        metadata: dict[str, Any]) -> None:
        """覆盖更新语义记忆内容（版本链由调用方另行写入）。"""
        with self._lock:
            self._conn.execute("UPDATE semantic_memories SET kind = ?, statement = ?, confidence = ?, evidence_ids = ?, updated_at = ?, version = ?, embedding = ?, embedding_dim = ?, metadata = ? WHERE id = ?", (kind, statement, confidence, self.dumps(evidence_ids), updated_at, version, embedding, embedding_dim, self.dumps(metadata), semantic_id))
            self._conn.commit()

    def touch_semantic(self, semantic_id: int, when_iso: str) -> None:
        """语义记忆的再激活簿记。"""
        with self._lock:
            self._conn.execute("UPDATE semantic_memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (when_iso, semantic_id))
            self._conn.commit()

    def set_semantic_status(self, semantic_id: int, status: str) -> bool:
        """迁移语义记忆的生命周期状态。"""
        with self._lock:
            cur = self._conn.execute("UPDATE semantic_memories SET status = ? WHERE id = ?", (status, semantic_id))
            self._conn.commit()
            return cur.rowcount > 0

    def semantic_count(self) -> int:
        """语义记忆总数。"""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM semantic_memories").fetchone()
        return int(row["n"])

    # -- semantic FTS ------------------------------------------------------------

    def semantic_fts_insert(self, rowid: int, statement: str, concept: str) -> None:
        """向语义 FTS 表写入一行文档。"""
        with self._lock:
            self._conn.execute("INSERT INTO semantic_fts(rowid, statement, concept) VALUES (?, ?, ?)", (rowid, statement, concept))
            self._conn.commit()

    def semantic_fts_delete(self, rowid: int) -> None:
        """从语义 FTS 表删除一行。"""
        with self._lock:
            self._conn.execute("DELETE FROM semantic_fts WHERE rowid = ?", (rowid,))
            self._conn.commit()

    def semantic_fts_search(self, match_expr: str, limit: int) -> list[sqlite3.Row]:
        """语义全文检索：返回 (rowid, rank)；bm25 rank 越小越相关。"""
        with self._lock:
            return self._conn.execute("SELECT rowid, bm25(semantic_fts) AS rank FROM semantic_fts WHERE semantic_fts MATCH ? ORDER BY rank LIMIT ?", (match_expr, limit)).fetchall()

    # -- memory versions -----------------------------------------------------------

    def insert_memory_version(self, *, semantic_id: int, version: int, statement: str,
                              confidence: float, evidence_ids: list[int],
                              created_at: str, change_reason: str) -> None:
        """追加一行版本历史（每个版本状态各一行）。"""
        with self._lock:
            self._conn.execute("INSERT INTO memory_versions (semantic_id, version, statement, confidence, evidence_ids, created_at, change_reason) VALUES (?, ?, ?, ?, ?, ?, ?)", (semantic_id, version, statement, confidence, self.dumps(evidence_ids), created_at, change_reason))
            self._conn.commit()

    def list_memory_versions(self, semantic_id: int) -> list[sqlite3.Row]:
        """按版本号升序列出某语义记忆的全部历史。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_versions WHERE semantic_id = ? ORDER BY version", (semantic_id,)).fetchall()

    # -- consolidation state ----------------------------------------------------------

    def get_consolidation_state(self, value: str) -> sqlite3.Row | None:
        """读取某概念（折叠 value）的增量巩固游标。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM consolidation_state WHERE value = ?", (value,)).fetchone()

    def list_consolidation_states(self) -> list[sqlite3.Row]:
        """全部巩固游标（分组时的差量计算用）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM consolidation_state").fetchall()

    def upsert_consolidation_state(self, *, value: str, representative_kind: str,
                                   last_consolidated_at: str, episode_count: int,
                                   semantic_id: int | None) -> None:
        """写入/更新游标（按折叠 value 为键，与代表 kind 无关）。"""
        with self._lock:
            self._conn.execute("INSERT INTO consolidation_state (value, representative_kind, last_consolidated_at, episode_count, semantic_id) VALUES (?, ?, ?, ?, ?) ON CONFLICT(value) DO UPDATE SET representative_kind = excluded.representative_kind, last_consolidated_at = excluded.last_consolidated_at, episode_count = excluded.episode_count, semantic_id = excluded.semantic_id", (value, representative_kind, last_consolidated_at, episode_count, semantic_id))
            self._conn.commit()

    # -- conflicts ----------------------------------------------------------------

    def insert_conflict(self, *, semantic_id: int, kind: str, status: str,
                        old_version: int, statement_before: str | None,
                        trigger_episode_id: int | None, trigger_kind: str,
                        detected_at: str, metadata: dict[str, Any]) -> int:
        """写入一条冲突记录，返回自增 id。"""
        with self._lock:
            cur = self._conn.execute("INSERT INTO memory_conflicts (semantic_id, kind, status, old_version, statement_before, trigger_episode_id, trigger_kind, detected_at, resolution_version, resolved_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)", (semantic_id, kind, status, old_version, statement_before, trigger_episode_id, trigger_kind, detected_at, self.dumps(metadata)))
            self._conn.commit()
            return int(cur.lastrowid)

    def get_conflict(self, conflict_id: int) -> sqlite3.Row | None:
        """按 id 取冲突记录。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_conflicts WHERE id = ?", (conflict_id,)).fetchone()

    def conflicts_for_semantic(self, semantic_id: int) -> list[sqlite3.Row]:
        """某语义记忆的全部冲突记录（按检测时间排序）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_conflicts WHERE semantic_id = ? ORDER BY detected_at", (semantic_id,)).fetchall()

    def open_conflict_semantic_ids(self) -> list[int]:
        """存在 open 冲突的语义记忆 id 去重列表（强制排期用）。"""
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT semantic_id FROM memory_conflicts WHERE status = 'open'").fetchall()
        return [row["semantic_id"] for row in rows]

    def resolve_conflicts_for_semantic(self, semantic_id: int, *, kind: str,
                                       resolution_version: int, resolved_at: str,
                                       dismiss: bool = False) -> int:
        """结清某语义记忆的全部 open 冲突。

        resolve 会记录最终分类（*kind*）；dismiss 保留原 kind 仅关闭记录。
        返回触及的冲突数。
        """
        with self._lock:
            if dismiss:
                cur = self._conn.execute("UPDATE memory_conflicts SET status = 'dismissed', resolution_version = ?, resolved_at = ? WHERE semantic_id = ? AND status = 'open'", (resolution_version, resolved_at, semantic_id))
            else:
                cur = self._conn.execute("UPDATE memory_conflicts SET kind = ?, status = 'resolved', resolution_version = ?, resolved_at = ? WHERE semantic_id = ? AND status = 'open'", (kind, resolution_version, resolved_at, semantic_id))
            self._conn.commit()
            return cur.rowcount

    def count_open_conflicts(self) -> int:
        """open 冲突总数。"""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM memory_conflicts WHERE status = 'open'").fetchone()
        return int(row["n"])

    def list_conflicts(self) -> list[sqlite3.Row]:
        """全部冲突记录（检测时间倒序）。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_conflicts ORDER BY detected_at DESC").fetchall()

    def list_conflicts_by_status(self, status: str) -> list[sqlite3.Row]:
        """按状态过滤冲突记录。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_conflicts WHERE status = ? ORDER BY detected_at DESC", (status,)).fetchall()

    def count_episodes_by_day(self, cutoff_iso: str) -> list[sqlite3.Row]:
        """截止时间之后，按天分组的编码数（Inspector 时间线）。"""
        with self._lock:
            return self._conn.execute("SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n FROM episodes WHERE created_at >= ? GROUP BY day ORDER BY day", (cutoff_iso,)).fetchall()

    def count_semantics_by_day(self, cutoff_iso: str) -> list[sqlite3.Row]:
        """截止时间之后，按天分组的新增知识数。"""
        with self._lock:
            return self._conn.execute("SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n FROM semantic_memories WHERE created_at >= ? GROUP BY day ORDER BY day", (cutoff_iso,)).fetchall()

    def count_conflicts_by_day(self, cutoff_iso: str) -> list[sqlite3.Row]:
        """截止时间之后，按天分组的冲突数。"""
        with self._lock:
            return self._conn.execute("SELECT substr(detected_at, 1, 10) AS day, COUNT(*) AS n FROM memory_conflicts WHERE detected_at >= ? GROUP BY day ORDER BY day", (cutoff_iso,)).fetchall()

    # -- explicit memory links (graph) -------------------------------------------

    def insert_link(self, *, source_kind: str, source_id: int, target_kind: str,
                    target_id: int, relation: str, weight: float,
                    created_by: str, created_at: str, metadata: dict[str, Any]) -> int:
        """插入一条显式关系边，返回自增 id。"""
        with self._lock:
            cur = self._conn.execute("INSERT INTO memory_links (source_kind, source_id, target_kind, target_id, relation, weight, created_by, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (source_kind, source_id, target_kind, target_id, relation, weight, created_by, created_at, self.dumps(metadata)))
            self._conn.commit()
            return int(cur.lastrowid)

    def get_link(self, link_id: int) -> sqlite3.Row | None:
        """按 id 取显式关系边。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_links WHERE id = ?", (link_id,)).fetchone()

    def delete_link(self, link_id: int) -> bool:
        """删除一条显式关系边。"""
        with self._lock:
            cur = self._conn.execute("DELETE FROM memory_links WHERE id = ?", (link_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def links_for_source(self, kind: str, node_id: int) -> list[sqlite3.Row]:
        """以该节点为源的全部显式边。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_links WHERE source_kind = ? AND source_id = ?", (kind, node_id)).fetchall()

    def links_for_target(self, kind: str, node_id: int) -> list[sqlite3.Row]:
        """以该节点为目标的全部显式边。"""
        with self._lock:
            return self._conn.execute("SELECT * FROM memory_links WHERE target_kind = ? AND target_id = ?", (kind, node_id)).fetchall()

    # -- grouping ---------------------------------------------------------------------

    def tag_group_counts(self) -> list[sqlite3.Row]:
        """按 (标签 kind, value) 统计活跃 episode 数——巩固的候选组。"""
        with self._lock:
            return self._conn.execute("SELECT t.kind AS kind, t.value AS value, COUNT(*) AS n FROM episode_tags t JOIN episodes e ON e.id = t.episode_id WHERE e.status = 'active' GROUP BY t.kind, t.value").fetchall()

    def concept_co_occurrence(self, value: str, limit: int) -> list[sqlite3.Row]:
        """活跃 episode 上与 *value* 共现的概念（entity/topic 按 value 折叠）。"""
        with self._lock:
            return self._conn.execute("SELECT a.value AS other, COUNT(*) AS n FROM episode_tags a JOIN episode_tags b ON a.episode_id = b.episode_id JOIN episodes e ON e.id = a.episode_id WHERE b.value = ? AND a.value != ? AND e.status = 'active' GROUP BY a.value ORDER BY n DESC, a.value LIMIT ?", (value, value, limit)).fetchall()

    # -- transactions -----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """把多次写入并入一次提交；出错回滚。

        仅在本块内直接使用原始 ``_conn``；上面的公开方法各自持锁。
        """
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        """关闭连接。"""
        with self._lock:
            self._conn.close()
