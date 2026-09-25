from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from brain_memory.consolidation.heuristic import HeuristicConsolidator
from brain_memory.models import Episode, SemanticKind


def _episode(episode_id: int, content: str, entities=None, topics=None) -> Episode:
    return Episode(
        id=episode_id,
        content=content,
        content_hash=f"h{episode_id}",
        entities=entities or [],
        topics=topics or [],
        created_at=datetime.now(timezone.utc) - timedelta(days=episode_id),
        importance=0.5,
        confidence=0.6,
    )


def test_co_occurrence_statement_zh_with_topics():
    episodes = [
        _episode(1, "用户在学习 Java 后端", entities=["Java"], topics=["java", "spring", "后端"]),
        _episode(2, "用户用 Java 写工具", entities=["Java"], topics=["java", "工具"]),
        _episode(3, "Java 项目重构", entities=["Java"], topics=["java", "重构"]),
    ]
    proposal = HeuristicConsolidator().propose("java", episodes, min_support=3)
    assert proposal is not None
    assert proposal.kind is SemanticKind.CO_OCCURRENCE
    assert "java" in proposal.statement and "3 条记忆" in proposal.statement
    assert proposal.supporting_indexes == [0, 1, 2]
    assert proposal.confidence <= 0.55


def test_confidence_capped_at_055_regardless_of_support():
    consolidator = HeuristicConsolidator()
    episodes = [_episode(i, f"Java thing {i}", entities=["java"]) for i in range(1, 12)]
    proposal = consolidator.propose("java", episodes, min_support=3)
    assert proposal.confidence == 0.55


def test_insufficient_support_returns_none():
    episodes = [_episode(1, "java alone", entities=["java"])]
    assert HeuristicConsolidator().propose("java", episodes, min_support=3) is None


def test_support_requires_concept_presence():
    episodes = [
        _episode(1, "java", entities=["java"]),
        _episode(2, "unrelated quantum chemistry"),
        _episode(3, "java again", entities=["java"]),
    ]
    proposal = HeuristicConsolidator().propose("java", episodes, min_support=3)
    assert proposal is None  # only 2 of 3 actually mention java


def test_english_statement_for_english_content():
    episodes = [
        _episode(i, f"Working on Rust async runtime day {i}", entities=["rust"], topics=["rust"])
        for i in (1, 2, 3)
    ]
    proposal = HeuristicConsolidator().propose("rust", episodes, min_support=3)
    assert proposal is not None
    assert "shows up across 3 memories" in proposal.statement
