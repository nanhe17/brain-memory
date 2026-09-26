"""检索指标。全部作用于排序后的记忆标签（最优在前）——
``"e<id>"`` 表示情景记忆，``"s<id>"`` 表示语义记忆——两套 id 空间
永不冲突。"""

from __future__ import annotations


def recall_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """Recall@k = |期望 ∩ top-k| / |期望|；没有期望项时视为满分。"""
    if not expected:
        return 1.0
    top = set(ranked[:k])
    return len(expected & top) / len(expected)


def precision_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """Precision@k = |期望 ∩ top-k| / k。"""
    if k <= 0:
        return 0.0
    return len(expected & set(ranked[:k])) / k


def mrr(expected: set[str], ranked: list[str]) -> float:
    """MRR = 1 / 第一个相关命中的名次；整榜无命中为 0。"""
    for position, memory_tag in enumerate(ranked, start=1):
        if memory_tag in expected:
            return 1.0 / position
    return 0.0


def violation_count(forbidden: set[str], ranked: list[str], k: int) -> int:
    """出现在 top-k 中的禁用记忆数量。"""
    return len(forbidden & set(ranked[:k]))
