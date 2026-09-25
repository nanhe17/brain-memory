from __future__ import annotations

import pytest

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine


def _engine(tmp_path, **overrides) -> MemoryEngine:
    # small candidate pool: the vector/keyword channels only return the very
    # best matches, so graph expansion is the only way weaker-related
    # memories (evidence, siblings) can surface — the regime where expansion
    # actually matters (small pools mimic large stores)
    return MemoryEngine(MemoryConfig(db_path=str(tmp_path / "exp.db"),
                                     embedding_dim=64,
                                     candidate_pool_per_channel=1,
                                     **overrides))


def _seed_knowledge(tmp_path, **overrides) -> MemoryEngine:
    engine = _engine(tmp_path, **overrides)
    engine.encode("用户在学习 Java 后端，正在读 Spring 源码")
    engine.encode("用户喜欢自学 Java 原理")
    engine.encode("用户整理了 Java 知识图谱")
    engine.consolidate()
    return engine


def _expanded(hits):
    return [h for h in hits if h.expanded]


def test_evidence_expansion_pulls_cited_episodes(tmp_path):
    engine = _seed_knowledge(tmp_path)
    # pool=2: base recall keeps the semantic + its best episode; the graph
    # must surface the remaining cited evidence with provenance
    hits = engine.recall("Java 学习总结", k=5)
    expanded = _expanded(hits)
    assert expanded, "evidence of a recalled semantic must be expanded in"
    for hit in expanded:
        assert hit.reasons and hit.reasons[0].startswith("graph:")
        assert hit.expanded is True
    anchors = [h for h in hits if not h.expanded]
    penalty = engine.config.expansion_penalty
    for hit in expanded:
        # every expanded entry sits below its would-be unpenalized score
        assert hit.score <= max(a.score for a in anchors) * penalty + 1e-9
    engine.close()


def test_sibling_expansion_via_shared_entity(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户完成了 Minecraft 飞机模组的机身建模")
    engine.encode("模组的涂装系统上线了")
    hits = engine.recall("机身设计进度", k=5, touch=False)
    expanded = _expanded(hits)
    assert expanded, "siblings sharing the entity must be expanded in"
    assert any(
        "shares entity" in reason
        for hit in expanded
        for reason in hit.reasons
    )
    engine.close()


def test_top_result_never_displaced_by_expansion(tmp_path):
    engine = _seed_knowledge(tmp_path)
    baseline = engine.recall("Java 学习总结", k=1, touch=False)
    hits = engine.recall("Java 学习总结", k=5, touch=False)
    assert hits[0].episode.id == baseline[0].episode.id
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    engine.close()


def test_expansion_respects_limit(tmp_path):
    engine = _seed_knowledge(tmp_path, expansion_limit=1)
    hits = engine.recall("Java 学习总结", k=5)
    assert len(_expanded(hits)) <= 1
    engine.close()


def test_expansion_can_be_disabled(tmp_path):
    engine = _seed_knowledge(tmp_path, recall_expansion=False)
    hits = engine.recall("Java 学习总结", k=5)
    assert _expanded(hits) == []
    engine.close()


def test_expansion_respects_recall_filters(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户在学习 Java 后端，正在读 Spring 源码", source="note")
    engine.encode("用户喜欢自学 Java 原理", source="note")
    engine.encode("用户整理了 Java 知识图谱", source="note")
    engine.consolidate()
    hits = engine.recall("Java 学习总结", k=5, source="conversation")
    assert all(h.episode.source == "conversation" for h in hits)
    engine.close()


def test_expanded_evidence_gets_reactivation_credit(tmp_path):
    engine = _seed_knowledge(tmp_path)
    before = {e.id: e.access_count for e in
              (engine._store.get(i) for i in (1, 2, 3))}
    engine.recall("Java 学习总结", k=2)
    after = {e.id: engine._store.get(i).access_count for e, i in
             zip((engine._store.get(i) for i in (1, 2, 3)), (1, 2, 3))}
    assert any(after[i] > before[i] for i in (1, 2, 3)), \
        "expanded-in evidence counts as a touch (reactivation, doc §15)"
    engine.close()
