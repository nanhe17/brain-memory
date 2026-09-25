from __future__ import annotations

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.models import ConflictKind, ConflictStatus


def _engine(tmp_path) -> MemoryEngine:
    return MemoryEngine(MemoryConfig(db_path=str(tmp_path / "challenge.db"), embedding_dim=64))


def _consolidated_engine(tmp_path) -> MemoryEngine:
    engine = _engine(tmp_path)
    engine.encode("用户在学习 Java 后端，正在读 Spring 源码")
    engine.encode("用户喜欢自学 Java 原理")
    engine.encode("用户整理了 Java 知识图谱")
    engine.consolidate()
    return engine


def test_correction_challenges_matching_concept(tmp_path):
    engine = _consolidated_engine(tmp_path)
    before = engine.find_semantic("java")

    result = engine.encode("其实我现在不想做 Java 了，打算转 AI")
    assert result.challenge is not None
    assert result.challenge.semantic_id == before.id
    assert result.challenge.status is ConflictStatus.OPEN
    assert result.challenge.kind is ConflictKind.UNRESOLVED
    assert result.challenge.trigger_episode_id == result.episode.id
    assert result.challenge.statement_before == before.statement

    # the challenge itself must not edit the belief
    after = engine.find_semantic("java")
    assert after.statement == before.statement
    assert after.confidence == before.confidence
    assert after.version == before.version
    engine.close()


def test_no_challenge_without_correction_signal(tmp_path):
    engine = _consolidated_engine(tmp_path)
    result = engine.encode("用户今天写了 Java 代码")
    assert result.challenge is None
    assert engine.conflicts(status="open") == []
    engine.close()


def test_no_challenge_without_matching_semantic(tmp_path):
    engine = _engine(tmp_path)
    result = engine.encode("其实我不喜欢量子化学了")
    assert result.challenge is None
    engine.close()


def test_indirect_correction_targets_sole_recalled_semantic(tmp_path):
    engine = _consolidated_engine(tmp_path)
    engine.recall("Java 学习", k=5)  # working memory now holds the semantic hit

    # correction without mentioning Java at all
    result = engine.encode("其实我改主意了，这个方向不做了")
    assert result.challenge is not None
    assert result.challenge.semantic_id == engine.find_semantic("java").id
    engine.close()


def test_indirect_correction_ambiguous_recall_is_ignored(tmp_path):
    engine = _consolidated_engine(tmp_path)
    engine.encode("用户在学习 Rust 异步编程")
    engine.encode("用户喜欢用 Rust 写工具")
    engine.encode("用户在研究 Rust 的 tokio 运行时")
    engine.consolidate()
    engine.recall("编程语言学习", k=5)  # recall holds 2+ semantic hits
    assert len(engine.conflicts(status="open")) == 0

    result = engine.encode("其实我改主意了，这个方向不做了")
    assert result.challenge is None
    engine.close()


def test_open_conflict_forces_reconsolidation(tmp_path):
    engine = _consolidated_engine(tmp_path)
    engine.encode("其实我现在不想做 Java 了，打算转 AI")   # challenge, no natural new-evidence need
    # the java group has +1 episode anyway; force the path through a manual
    # reconsolidate with no additional episodes to prove forcing works
    report = engine.reconsolidate(engine.find_semantic("java").id)
    assert report.updated, "open conflict must force the group due"
    assert engine.conflicts(status="open") == []
    engine.close()


def test_stats_and_inspect_surface_conflicts(tmp_path):
    engine = _consolidated_engine(tmp_path)
    engine.encode("其实我现在不想做 Java 了")
    assert engine.stats().open_conflicts == 1

    inspection = engine.inspect_semantic(engine.find_semantic("java").id)
    assert len(inspection["conflicts"]) == 1
    assert inspection["conflicts"][0].status is ConflictStatus.OPEN
    engine.close()
