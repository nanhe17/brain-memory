from __future__ import annotations

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.models import SemanticKind


def _engine(tmp_path) -> MemoryEngine:
    config = MemoryConfig(
        db_path=str(tmp_path / "cons.db"),
        embedding_dim=64,
        consolidation_min_support=3,
    )
    return MemoryEngine(config)


def _seed_learning_episodes(engine) -> None:
    engine.encode("用户在学习 Java 后端，正在读 Spring 的源码")
    engine.encode("用户说不想只看课程，更喜欢自学原理，最近在学 Java")
    engine.encode("用户整理了 Java 的知识图谱，把 Spring 的 bean 串了起来")


def test_consolidation_creates_semantic_memory_with_versions(tmp_path):
    engine = _engine(tmp_path)
    _seed_learning_episodes(engine)

    report = engine.consolidate()
    assert report.groups_considered >= 1
    assert len(report.created) == 1
    memory = report.created[0]
    assert memory.concept == "java"
    assert memory.kind is SemanticKind.CO_OCCURRENCE
    assert memory.version == 1
    assert len(memory.evidence_ids) == 3
    assert memory.confidence <= 0.55  # deterministic proposals never claim more

    inspection = engine.inspect_semantic(memory.id)
    assert len(inspection["evidence"]) == 3
    assert [v.version for v in inspection["versions"]] == [1]
    engine.close()


def test_new_evidence_bumps_version_and_merges_evidence(tmp_path):
    engine = _engine(tmp_path)
    _seed_learning_episodes(engine)
    first = engine.consolidate().created[0]

    engine.encode("用户又用 Java 写了一个爬虫练习")
    report = engine.consolidate()
    assert not report.created
    assert len(report.updated) == 1
    updated = report.updated[0]
    assert updated.id == first.id
    assert updated.version == 2
    assert len(updated.evidence_ids) == 4

    versions = engine.inspect_semantic(first.id)["versions"]
    assert [v.version for v in versions] == [1, 2]
    engine.close()


def test_consolidation_is_idempotent_without_new_evidence(tmp_path):
    engine = _engine(tmp_path)
    _seed_learning_episodes(engine)
    engine.consolidate()
    second = engine.consolidate()
    assert second.groups_considered == 0
    assert not second.created and not second.updated
    assert engine.stats().semantic_memories == 1
    engine.close()


def test_min_support_blocks_tiny_groups(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户在学 Java")
    engine.encode("用户在聊 Java 的历史")
    report = engine.consolidate()
    assert not report.created
    assert report.skipped and "java" in report.skipped[0]
    engine.close()


def test_unified_recall_returns_semantic_hits(tmp_path):
    engine = _engine(tmp_path)
    _seed_learning_episodes(engine)
    memory = engine.consolidate().created[0]

    hits = engine.recall("Java 学习", k=5)
    semantic_hits = [h for h in hits if h.is_semantic]
    assert semantic_hits, "consolidated knowledge must be recallable"
    top_semantic = semantic_hits[0]
    assert top_semantic.semantic.id == memory.id
    assert top_semantic.episode.content == memory.statement  # view carries statement
    assert top_semantic.episode.importance == memory.confidence

    engine.close()


def test_semantic_recall_limit_caps_knowledge_hits(tmp_path):
    config = MemoryConfig(
        db_path=str(tmp_path / "cap.db"),
        embedding_dim=64,
        semantic_recall_limit=1,
    )
    engine = MemoryEngine(config)
    engine.encode("用户在学习 Java 后端和 Spring")
    engine.encode("用户喜欢自学 Java 原理")
    engine.encode("用户用 Java 写了学习笔记")
    engine.encode("用户在用 Rust 写 CLI 工具")
    engine.encode("用户用 Rust 重写了旧脚本")
    engine.encode("用户研究 Rust 的 async 运行时")
    engine.consolidate()
    assert engine.stats().semantic_memories == 2

    hits = engine.recall("编程语言学习", k=10)
    assert len([h for h in hits if h.is_semantic]) <= 1
    engine.close()


def test_prompt_block_renders_known_facts(tmp_path):
    engine = _engine(tmp_path)
    _seed_learning_episodes(engine)
    engine.consolidate()
    engine.recall("Java 学习", k=5)
    block = engine.working.build_prompt_block()
    assert "[KNOWN FACTS]" in block
    assert "S-" in block and "confidence" in block
    assert "[RELEVANT MEMORIES]" in block
    engine.close()


def test_llm_consolidation_failure_falls_back_to_heuristic(tmp_path):
    engine = _engine(tmp_path, )

    class _Boom:
        def propose(self, concept, episodes, *, min_support):
            raise RuntimeError("llm down")

    engine._consolidator.set_llm(_Boom())
    _seed_learning_episodes(engine)
    report = engine.consolidate()
    assert len(report.created) == 1
    assert report.created[0].kind is SemanticKind.CO_OCCURRENCE
    engine.close()


def test_llm_consolidation_used_when_configured(tmp_path):
    from brain_memory.models import PatternProposal, SemanticKind

    engine = _engine(tmp_path)

    class _FakeLLM:
        def propose(self, concept, episodes, *, min_support):
            return PatternProposal(
                concept=concept,
                statement="User is systematically self-learning the Java backend stack.",
                kind=SemanticKind.GENERALIZATION,
                confidence=0.85,
                supporting_indexes=[0, 1, 2],
            )

    engine._consolidator.set_llm(_FakeLLM())
    _seed_learning_episodes(engine)
    report = engine.consolidate()
    assert report.created[0].kind is SemanticKind.GENERALIZATION
    assert report.created[0].confidence == 0.85
    engine.close()
