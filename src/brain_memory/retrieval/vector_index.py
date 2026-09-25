"""In-process brute-force vector search.

For the scales this phase targets (up to ~100k active episodes) a numpy dot
product over an in-memory matrix is millisecond-fast and dependency-free.
Embeddings are L2-normalized at write time, so cosine similarity is a single
matrix-vector product.  The index caches the matrix and reloads only when the
store signals new writes via :meth:`invalidate`.
"""

from __future__ import annotations

import numpy as np

from brain_memory.storage.db import Database


class VectorIndex:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._ids: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._dirty = True

    def invalidate(self) -> None:
        self._dirty = True

    def _load(self) -> None:
        rows = self._db.list_active_embeddings()
        if rows:
            ids = np.fromiter((row["id"] for row in rows), dtype=np.int64, count=len(rows))
            dim = rows[0]["embedding_dim"]
            matrix = np.stack([
                np.frombuffer(row["embedding"], dtype=np.float32, count=dim) for row in rows
            ])
        else:
            ids = np.empty(0, dtype=np.int64)
            matrix = np.empty((0, 0), dtype=np.float32)
        self._ids, self._matrix, self._dirty = ids, matrix, False

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Top-k ``(episode_id, cosine_similarity)`` pairs, best first."""
        if self._dirty or self._matrix is None or self._ids is None:
            self._load()
        assert self._matrix is not None and self._ids is not None
        if self._matrix.size == 0 or k <= 0:
            return []
        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"query dim {query.shape[0]} != index dim {self._matrix.shape[1]}; "
                "the embedding provider changed without re-encoding the store"
            )
        sims = self._matrix @ query
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            mask = np.isin(self._ids, np.fromiter(allowed_ids, dtype=np.int64, count=len(allowed_ids)))
            sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [
            (int(self._ids[i]), float(sims[i]))
            for i in order
            if np.isfinite(sims[i])
        ]
