"""嵌入 providers。

引擎只看得到 :class:`EmbeddingProvider` 协议。向量在写入时做 L2 归一化
（余弦相似度退化为点积），以 float32 BLOB 形式存入 SQLite。
"""

from brain_memory.embeddings.base import EmbeddingError, EmbeddingProvider, normalize_rows

__all__ = ["EmbeddingProvider", "EmbeddingError", "normalize_rows"]
