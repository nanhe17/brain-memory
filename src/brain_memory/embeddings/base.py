"""Embedding provider contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails irrecoverably."""


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero (they match nothing)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text -> fixed-dimension L2-normalized float32 vectors.

    ``dim`` may be ``None`` for providers that only learn the dimension from
    their first response (cloud APIs).  The engine asserts consistency across
    every batch.
    """

    name: str
    dim: int | None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 matrix, rows L2-normalized."""
        ...
