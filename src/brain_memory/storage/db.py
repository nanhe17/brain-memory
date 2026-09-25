"""SQLite storage layer (repository style).

Design notes
------------
* Every SQL statement is an inline literal executed with ``?`` placeholders
  only — no string formatting, concatenation, or variable SQL anywhere, so
  nothing caller-controlled can alter query structure.  Schema migrations are
  written out statement-by-statement for the same reason, and every column is
  written explicitly on insert (no DEFAULT clauses needed).
* Write paths are serialized behind a lock (single-writer assumption,
  documented for Phase 1; PostgreSQL can replace this later behind the same
  interface).
* FTS5 provides keyword search with zero extra dependencies.  The FTS table
  is synced manually by the episodic store (insert/delete).
* ``episode_tags`` is the entity/topic index used for metadata filtering and,
  later, for consolidation grouping.
* ``semantic_memories`` is created now (Phase 3 will write into it) so the
  schema does not churn once consolidation lands.
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
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thread-safe sqlite3 wrapper with sequential schema migrations.

    Data-access methods are named and narrow; SQL lives inline at each call
    site with bound parameters exclusively.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: every access goes through self._lock.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # -- migrations ---------------------------------------------------------

    def _migrate(self) -> None:
        """Apply schema migrations in order.

        Migration 1 creates the base schema.  Later migrations append a new
        ``_apply_migration_N`` method and a version branch here.
        """
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
            self._conn.commit()
            row = self._conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM schema_version").fetchone()
            current = row["v"]
            if current < 1:
                self._apply_migration_1()
                self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (1,))
                self._conn.commit()

    def _apply_migration_1(self) -> None:
        self._conn.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT NOT NULL, entities TEXT NOT NULL, topics TEXT NOT NULL, key_facts TEXT NOT NULL, emphasis TEXT NOT NULL, context TEXT, source TEXT NOT NULL, created_at TEXT NOT NULL, embedding BLOB, embedding_dim INTEGER NOT NULL, importance REAL NOT NULL, confidence REAL NOT NULL, access_count INTEGER NOT NULL, last_accessed TEXT, status TEXT NOT NULL, metadata TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(created_at)")
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(content, entities_text, topics_text)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS episode_tags (episode_id INTEGER NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_value ON episode_tags(kind, value)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS semantic_memories (id INTEGER PRIMARY KEY, concept TEXT NOT NULL, statement TEXT NOT NULL, confidence REAL NOT NULL, evidence_ids TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL, metadata TEXT NOT NULL)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")

    # -- json / datetime helpers --------------------------------------------

    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def loads(text: str | None, default: Any) -> Any:
        if not text:
            return default
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def parse_dt(text: str | None) -> datetime | None:
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
        with self._lock:
            cur = self._conn.execute("INSERT INTO episodes (content, content_hash, entities, topics, key_facts, emphasis, context, source, created_at, embedding, embedding_dim, importance, confidence, access_count, last_accessed, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 'active', ?)", (content, content_hash, self.dumps(entities), self.dumps(topics), self.dumps(key_facts), self.dumps(emphasis), context, source, created_at, embedding, embedding_dim, importance, confidence, self.dumps(metadata)))
            self._conn.commit()
            return int(cur.lastrowid)

    def get_episode(self, episode_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()

    def get_active_episode_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes WHERE content_hash = ? AND status = 'active'", (content_hash,)).fetchone()

    def touch_episode(self, episode_id: int, when_iso: str) -> None:
        """Reactivation bookkeeping: bump access stats (see design doc §15)."""
        with self._lock:
            self._conn.execute("UPDATE episodes SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (when_iso, episode_id))
            self._conn.commit()

    def set_episode_status(self, episode_id: int, status: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE episodes SET status = ? WHERE id = ?", (status, episode_id))
            self._conn.commit()
            return cur.rowcount > 0

    def list_active_episodes(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM episodes WHERE status = 'active' ORDER BY created_at DESC").fetchall()

    def list_active_embeddings(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT id, embedding, embedding_dim FROM episodes WHERE status = 'active' AND embedding IS NOT NULL").fetchall()

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM episodes GROUP BY status").fetchall()
        return {row["status"]: row["n"] for row in rows}

    def episode_extremes(self) -> sqlite3.Row:
        with self._lock:
            return self._conn.execute("SELECT MIN(created_at) AS oldest, MAX(created_at) AS newest, AVG(importance) AS avg_importance FROM episodes").fetchone()

    # -- tags (entity / topic index) ------------------------------------------

    def insert_tags(self, tags: list[tuple[int, str, str]]) -> None:
        if not tags:
            return
        with self._lock:
            self._conn.executemany("INSERT INTO episode_tags(episode_id, kind, value) VALUES (?, ?, ?)", tags)
            self._conn.commit()

    def delete_tags(self, episode_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM episode_tags WHERE episode_id = ?", (episode_id,))
            self._conn.commit()

    def episode_ids_by_tag(self, kind: str, value: str) -> list[int]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT episode_id FROM episode_tags WHERE kind = ? AND value = ?", (kind, value)).fetchall()
        return [row["episode_id"] for row in rows]

    # -- full-text search -------------------------------------------------------

    def fts_insert(self, rowid: int, content: str, entities_text: str, topics_text: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO episodes_fts(rowid, content, entities_text, topics_text) VALUES (?, ?, ?, ?)", (rowid, content, entities_text, topics_text))
            self._conn.commit()

    def fts_delete(self, rowid: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM episodes_fts WHERE rowid = ?", (rowid,))
            self._conn.commit()

    def fts_search(self, match_expr: str, limit: int) -> list[sqlite3.Row]:
        """Returns (rowid, rank) pairs; bm25 rank — smaller is better."""
        with self._lock:
            return self._conn.execute("SELECT rowid, bm25(episodes_fts) AS rank FROM episodes_fts WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?", (match_expr, limit)).fetchall()

    # -- transactions -----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Group several writes into one commit; rolls back on error.

        Only touch the raw connection inside this block; the public methods
        above take the lock themselves.
        """
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()
