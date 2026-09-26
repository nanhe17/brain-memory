"""进程内暴力向量检索。

对本阶段的目标规模（约 10 万条活跃 episode 以内），内存矩阵上的 numpy
点积是毫秒级且零依赖。嵌入在写入时已 L2 归一化，余弦相似度就是一次
矩阵-向量乘。索引缓存矩阵，仅在存储通知有新写入（:meth:`invalidate`）
时重载。
"""

from __future__ import annotations

import numpy as np

from brain_memory.storage.db import Database


class VectorIndex:
    """单一嵌入来源上的暴力索引。

    ``rows_provider`` 返回 ``(id, embedding, embedding_dim)`` 行——默认
    是情景库的活跃嵌入；语义库传入自己的加载器，两套 id 空间各自独立
    建索引。
    """

    def __init__(self, db: Database, rows_provider=None) -> None:
        self._db = db
        self._rows_provider = rows_provider or db.list_active_embeddings
        self._ids: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._dirty = True

    def invalidate(self) -> None:
        """标记缓存失效（新写入/状态迁移后由引擎调用）。"""
        self._dirty = True

    def _load(self) -> None:
        """从数据源加载 id 向量与矩阵。"""
        rows = self._rows_provider()
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
        """top-k ``(episode_id, cosine_similarity)`，最优在前。"""
        if self._dirty or self._matrix is None or self._ids is None:
            self._load()
        assert self._matrix is not None and self._ids is not None
        if self._matrix.size == 0 or k <= 0:
            return []
        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._matrix.shape[1]:
            # 维度不一致意味着 provider 换了但库没重编码——大声失败
            # 而不是静默错误排序
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
