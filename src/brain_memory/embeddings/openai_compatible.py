"""Embeddings from any OpenAI-compatible ``POST /embeddings`` endpoint.

Works with Zhipu GLM (base_url ``https://open.bigmodel.cn/api/paas/v4``,
model ``embedding-3``) and OpenAI (``https://api.openai.com/v1``,
``text-embedding-3-small`` etc.).  Dimension is learned from the first
response and then asserted on every call.
"""

from __future__ import annotations

import time

import httpx
import numpy as np

from brain_memory.embeddings.base import EmbeddingError, normalize_rows


class OpenAICompatibleEmbedder:
    """Small sync client with fixed retries; batches large text lists."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        batch_size: int = 32,
        retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAICompatibleEmbedder requires an API key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.retries = retries
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"openai_compatible:{self.model}"

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        matrix = np.asarray(vectors, dtype=np.float32)
        if self._dim is None:
            self._dim = int(matrix.shape[1])
        if matrix.shape[1] != self._dim:
            raise EmbeddingError(
                f"inconsistent embedding dim: got {matrix.shape[1]}, expected {self._dim}"
            )
        return normalize_rows(matrix)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()["data"]
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        raise EmbeddingError(f"embedding request failed after retries: {last_error}")
