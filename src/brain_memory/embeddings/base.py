"""嵌入 provider 契约。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class EmbeddingError(RuntimeError):
    """嵌入后端不可恢复的失败。"""


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """逐行 L2 归一化；零向量保持为零（与任何向量都不相似）。"""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本 -> 固定维度的 L2 归一化 float32 向量。

    ``dim`` 对云 provider 可以为 ``None``——维度只能从首次响应学习；
    引擎会对每个批次断言维度一致。
    """

    name: str
    dim: int | None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """返回 ``(len(texts), dim)`` 的 float32 矩阵，行已 L2 归一化。"""
        ...
