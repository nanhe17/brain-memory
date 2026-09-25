"""Embedding providers.

The engine only ever sees the :class:`EmbeddingProvider` protocol.  Vectors
are L2-normalized so cosine similarity is a dot product, and stored as
float32 BLOBs in SQLite.
"""

from brain_memory.embeddings.base import EmbeddingError, EmbeddingProvider, normalize_rows

__all__ = ["EmbeddingProvider", "EmbeddingError", "normalize_rows"]
