"""任意 OpenAI 兼容 ``POST /embeddings`` 端点的嵌入客户端。

适配智谱 GLM（base_url ``https://open.bigmodel.cn/api/paas/v4``，模型
``embedding-3``）与 OpenAI（``https://api.openai.com/v1``、
``text-embedding-3-small`` 等）。维度从首次响应学习，之后每次调用断言。
"""

from __future__ import annotations

import time

import httpx
import numpy as np

from brain_memory.embeddings.base import EmbeddingError, normalize_rows


class OpenAICompatibleEmbedder:
    """小型同步客户端，固定重试次数；大批文本自动分批。"""

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
        """provider 名称（含模型名）。"""
        return f"openai_compatible:{self.model}"

    @property
    def dim(self) -> int | None:
        """向量维度（首次响应前为 None）。"""
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """分批请求嵌入并拼装归一化矩阵。"""
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        matrix = np.asarray(vectors, dtype=np.float32)
        # 首次响应学习维度，之后每次调用断言一致
        if self._dim is None:
            self._dim = int(matrix.shape[1])
        if matrix.shape[1] != self._dim:
            raise EmbeddingError(
                f"inconsistent embedding dim: got {matrix.shape[1]}, expected {self._dim}"
            )
        return normalize_rows(matrix)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """单批请求，指数退避重试；最终失败抛 EmbeddingError。"""
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
