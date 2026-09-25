"""Retrieval metrics.  All operate on ranked episode ids (best first)."""

from __future__ import annotations


def recall_at_k(expected: set[int], ranked: list[int], k: int) -> float:
    """|expected ∩ top-k| / |expected|; 1.0 when nothing was expected."""
    if not expected:
        return 1.0
    top = set(ranked[:k])
    return len(expected & top) / len(expected)


def precision_at_k(expected: set[int], ranked: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(expected & set(ranked[:k])) / k


def mrr(expected: set[int], ranked: list[int]) -> float:
    """1 / rank of the first relevant hit; 0.0 if none in the whole ranking."""
    for position, episode_id in enumerate(ranked, start=1):
        if episode_id in expected:
            return 1.0 / position
    return 0.0


def violation_count(forbidden: set[int], ranked: list[int], k: int) -> int:
    """How many forbidden memories surfaced in the top-k."""
    return len(forbidden & set(ranked[:k]))
