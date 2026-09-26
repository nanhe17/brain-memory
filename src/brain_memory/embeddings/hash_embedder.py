"""确定性特征哈希嵌入器。

语义质量刻意*不是*目标——这个 provider 的存在意义是让整条管线（编码、
存储、检索、排序、测试、演示）离线、可复现、零依赖、零 API key 地运行。
词元身份被哈希进桶；词汇重叠产生相似度，其余不产生。生产质量请换用
真实嵌入 provider。
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from brain_memory.embeddings.base import normalize_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    """分词：拉丁单词 + 中文字符与二元组。"""
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    cjk_chars = _CJK_RE.findall(lowered)
    tokens.extend(cjk_chars)
    tokens.extend(f"{a}{b}" for a, b in zip(cjk_chars, cjk_chars[1:]))
    return tokens


class HashEmbedder:
    """稳定的特征哈希向量：相同文本永远得到相同行。"""

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("hash embedding dim must be >= 16")
        self._dim = dim

    @property
    def name(self) -> str:
        """provider 名称（含维度）。"""
        return f"hash:{self._dim}"

    @property
    def dim(self) -> int:
        """向量维度。"""
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """逐文本特征哈希：桶累加（符号由哈希决定），行归一化。"""
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
