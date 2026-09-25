"""Retrieval metrics.  All operate on ranked memory tags (best first) —
``"e<id>"`` for episodic memories, ``"s<id>"`` for semantic ones — so episode
and semantic id spaces never collide."""


from __future__ import annotations


def recall_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """|expected ∩ top-k| / |expected|; 1.0 when nothing was expected."""
    if not expected:
        return 1.0
    top = set(ranked[:k])
    return len(expected & top) / len(expected)


def precision_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(expected & set(ranked[:k])) / k


def mrr(expected: set[str], ranked: list[str]) -> float:
    """1 / rank of the first relevant hit; 0.0 if none in the whole ranking."""
    for position, memory_tag in enumerate(ranked, start=1):
        if memory_tag in expected:
            return 1.0 / position
    return 0.0


def violation_count(forbidden: set[str], ranked: list[str], k: int) -> int:
    """How many forbidden memories surfaced in the top-k."""
    return len(forbidden & set(ranked[:k]))
