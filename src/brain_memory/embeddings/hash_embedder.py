"""Deterministic feature-hashing embedder.

Semantic quality is intentionally *not* the goal — this provider exists so
the whole pipeline (encoding, storage, retrieval, ranking, tests, demos)
runs offline and reproducibly with zero dependencies and zero API keys.
Token identity is hashed into buckets; overlapping vocabulary yields
similarity, everything else does not.  Swap in a real embedding provider for
production quality.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from brain_memory.embeddings.base import normalize_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    cjk_chars = _CJK_RE.findall(lowered)
    tokens.extend(cjk_chars)
    tokens.extend(f"{a}{b}" for a, b in zip(cjk_chars, cjk_chars[1:]))
    return tokens


class HashEmbedder:
    """Stable feature-hashing vectors; same text always yields the same row."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("hash embedding dim must be >= 16")
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hash:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in _tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self._dim
                sign = 1.0 if digest[8] & 1 else -1.0
                counts[bucket] = counts.get(bucket, 0.0) + sign
            for bucket, weight in counts.items():
                rows[i, bucket] = weight
        return normalize_rows(rows)
