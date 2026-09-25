from __future__ import annotations

import pytest

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.models import ConflictKind, PatternProposal, SemanticKind


def _engine(tmp_path, **overrides) -> MemoryEngine:
    config = MemoryConfig(
        db_path=str(tmp_path / "recon.db"), embedding_dim=64, **overrides
    )
    return MemoryEngine(config)


def _seed(engine: MemoryEngine) -> None:
    engine.encode("用户喜欢 Java，每天写 Spring 项目")
    engine.encode("用户在深入学习 Java 虚拟机调优")
    engine.encode("用户用 Java 完成了一个大项目")


def test_polarity_flip_produces_evolution_with_temporal_narrative(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine)
    first = engine.consolidate().created[0]
    assert first.metadata.get("polarity") == {"pos": 1, "neg": 0, "mixed": 0}

    engine.encode("用户说其实不想做 Java 了，打算转 AI 方向")
    report = engine.consolidate()
    assert len(report.updated) == 1
    updated = report.updated[0]
    assert updated.version == 2
    assert updated.metadata["polarity"] == {"pos": 1, "neg": 1, "mixed": 0}

    conflict = report.conflicts[0]
    assert conflict.kind is ConflictKind.EVOLUTION
    assert conflict.status.value == "resolved"
    assert conflict.old_version == 1
    assert conflict.statement_before == first.statement
    assert conflict.resolution_version == 2

    # temporal narrative preserved in the statement and the version trail
    assert "变化" in updated.statement or "shifted" in updated.statement
    versions = engine.inspect_semantic(first.id)["versions"]
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].statement == first.statement
    engine.close()


def test_no_polarity_change_dismisses_open_challenges(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine)
    engine.consolidate()
    semantic_id = engine.find_semantic("java").id

    # a correction challenge that reconsolidation cannot substantiate
    engine.encode("其实我对 Java 的看法有点变化")
    assert engine.conflicts(status="open")

    report = engine.reconsolidate(semantic_id)
    assert report.updated  # forced run, consistent refinement
    assert engine.conflicts(status="open") == []
    dismissed = [c for c in engine.conflicts() if c.status.value == "dismissed"]
    assert dismissed, "unsubstantiated challenge must be dismissed, not resolved"
    engine.close()


def test_reconsolidation_threshold_gates_llm_rewrite(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine)
    engine.consolidate()
    semantic_id = engine.find_semantic("java").id

    class _WeakLLM:
        def propose(self, concept, episodes, *, min_support):
            return None

        def repropose(self, concept, episodes, existing, *, min_support):
            # claims a contradiction supported by only ONE episode
            return PatternProposal(
                concept=concept,
                statement="User has abandoned Java entirely.",
                kind=SemanticKind.FACT,
                confidence=0.9,
                supporting_indexes=[0],
                change_kind="contradiction",
            )

    engine._consolidator.set_llm(_WeakLLM())
    engine.encode("其实我不太想做 Java 了")
    report = engine.reconsolidate(semantic_id)
    assert not report.created and not report.updated
    assert engine.find_semantic("java").version == 1  # old belief untouched
    assert engine.conflicts(status="open"), "gate failure keeps the challenge open"
    engine.close()


def test_llm_repropose_temporal_narrative(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine)
    engine.consolidate()

    class _NarrativeLLM:
        def propose(self, concept, episodes, *, min_support):
            return None

        def repropose(self, concept, episodes, existing, *, min_support):
            return PatternProposal(
                concept=concept,
                statement="用户过去长期深入 Java 后端，但近期兴趣已转向 AI 方向。",
                kind=SemanticKind.GENERALIZATION,
                confidence=0.8,
                supporting_indexes=[0, 1],
                change_kind="evolution",
            )

    engine.encode("其实我现在更想做 AI 了")
    engine._consolidator.set_llm(_NarrativeLLM())
    report = engine.reconsolidate(engine.find_semantic("java").id)
    updated = report.updated[0]
    assert "过去" in updated.statement and "已" in updated.statement
    assert report.conflicts[0].kind is ConflictKind.EVOLUTION
    assert updated.kind is SemanticKind.GENERALIZATION  # LLM kind is honored
    engine.close()


def test_phase3_behavior_unchanged_without_polarity_baseline(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine)
    first = engine.consolidate().created[0]
    engine.encode("用户又用 Java 写了一个爬虫练习")
    report = engine.consolidate()
    updated = report.updated[0]
    assert updated.version == 2
    assert updated.statement != first.statement  # normal refinement
    assert report.conflicts == []  # no polarity baseline shift -> consistent
    engine.close()


def test_manual_reconsolidate_unknown_id(tmp_path):
    engine = _engine(tmp_path)
    with pytest.raises(ValueError):
        engine.reconsolidate(9999)
    engine.close()


def test_llm_repropose_transport_failure_falls_back(tmp_path):
    import httpx

    engine = _engine(tmp_path)
    _seed(engine)
    engine.consolidate()

    class _BoomLLM:
        def propose(self, concept, episodes, *, min_support):
            return None

        def repropose(self, concept, episodes, existing, *, min_support):
            raise httpx.ConnectError("down")

    engine.encode("其实我不想做 Java 了，转 AI 去了")
    engine._consolidator.set_llm(_BoomLLM())
    report = engine.reconsolidate(engine.find_semantic("java").id)
    assert report.updated  # deterministic polarity path took over
    assert report.conflicts[0].kind is ConflictKind.EVOLUTION
    engine.close()
