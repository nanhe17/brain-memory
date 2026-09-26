from __future__ import annotations

from datetime import datetime, timezone

import pytest

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.graph.ppr import personalized_pagerank


def _engine(tmp_path, **overrides) -> MemoryEngine:
    return MemoryEngine(MemoryConfig(db_path=str(tmp_path / "ppr.db"),
                                     embedding_dim=64, **overrides))


def _consolidated(tmp_path) -> MemoryEngine:
    engine = _engine(tmp_path)
    engine.encode("用户在学习 Java 后端，正在读 Spring 源码")
    engine.encode("用户喜欢自学 Java 原理")
    engine.encode("用户整理了 Java 知识图谱")
    engine.consolidate()
    return engine


# -- algorithm properties (hand-checked on the seeded store) -----------------------


def test_mass_conservation_and_determinism(tmp_path):
    engine = _consolidated(tmp_path)
    seeds = {"c:java": 1.0}
    first = personalized_pagerank(
        engine._db, engine._store, engine._semantic_store, seeds=seeds
    )
    second = personalized_pagerank(
        engine._db, engine._store, engine._semantic_store, seeds=seeds
    )
    assert first == second  # deterministic
    assert pytest.approx(sum(first.values()), abs=1e-6) == 1.0  # mass conserved
    engine.close()


def test_seed_nodes_dominate_unreachable_are_zero(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户在学习 Java 后端")       # java cluster
    engine.encode("用户在研究量子化学")          # unreachable from java seed
    seeds = {"c:java": 1.0}
    mass = personalized_pagerank(
        engine._db, engine._store, engine._semantic_store, seeds=seeds
    )
    java_mass = mass.get("c:java", 0.0)
    chem_mass = mass.get("c:量子化学", 0.0)
    assert java_mass > 0.1
    assert chem_mass < 1e-9  # unreachable: no path from java
    # the seed concept outranks every concept it cannot reach
    other_concepts = {k: v for k, v in mass.items()
                      if k.startswith("c:") and k != "c:java"}
    assert all(java_mass >= v for v in other_concepts.values())
    engine.close()


def test_evidence_semantics_receive_mass(tmp_path):
    engine = _consolidated(tmp_path)
    mass = personalized_pagerank(
        engine._db, engine._store, engine._semantic_store, seeds={"c:java": 1.0}
    )
    semantic_mass = mass.get("s1", 0.0)
    assert semantic_mass > 0.0, "consolidated knowledge sits on the graph"
    episodes = [v for k, v in mass.items() if k.startswith("e")]
    assert max(episodes) > semantic_mass or semantic_mass > 0.05
    engine.close()


def test_damping_changes_spread(tmp_path):
    engine = _consolidated(tmp_path)
    low_spread = personalized_pagerank(
        engine._db, engine._store, engine._semantic_store,
        seeds={"c:java": 1.0}, damping=0.5,
    )
    high_spread = personalized_pagerank(
        engine._db, engine._store, engine._semantic_store,
        seeds={"c:java": 1.0}, damping=0.95,
    )
    # higher damping pushes more mass off the seed into the neighborhood
    assert high_spread["c:java"] < low_spread["c:java"]
    engine.close()


def test_empty_seeds_returns_empty(tmp_path):
    engine = _consolidated(tmp_path)
    assert personalized_pagerank(
        engine._db, engine._store, engine._semantic_store, seeds={}
    ) == {}
    engine.close()


# -- engine integration ---------------------------------------------------------------


def test_graph_rank_standalone(tmp_path):
    engine = _consolidated(tmp_path)
    ranking = engine.graph_rank("Java 学习", k=5)
    assert ranking
    # the seed concept stays in the top of a descending-mass ranking
    refs = [ref for ref, _ in ranking]
    assert "c:java" in refs[:3]
    masses = [m for _, m in ranking]
    assert masses == sorted(masses, reverse=True)
    engine.close()


def test_graph_rank_without_concepts(tmp_path):
    engine = _consolidated(tmp_path)
    assert engine.graph_rank("嗯嗯啊啊") == []  # no lexicon concepts -> no seeds
    engine.close()


def test_ppr_blend_changes_scores_when_enabled(tmp_path):
    engine = _consolidated(tmp_path)
    baseline = {h.episode.id: h.score for h in engine.recall("Java 学习", k=5, touch=False)}

    engine.config = engine.config.model_copy(update={"graph_ppr": True, "ppr_mix": 0.5})
    blended = engine.recall("Java 学习", k=5, touch=False)
    blended_scores = {h.episode.id: h.score for h in blended}

    assert any(
        abs(blended_scores.get(eid, 0) - score) > 1e-9
        for eid, score in baseline.items()
    ), "PPR blend must move scores when enabled"
    ppr_reasons = [r for h in blended for r in h.reasons if r.startswith("graph-ppr")]
    assert ppr_reasons
    engine.close()


def test_ppr_off_by_default(tmp_path):
    engine = _consolidated(tmp_path)
    assert engine.config.graph_ppr is False
    hits = engine.recall("Java 学习", k=5)
    assert not [r for h in hits for r in h.reasons if r.startswith("graph-ppr")]
    engine.close()


def test_ppr_appends_structural_matches(tmp_path):
    engine = _engine(tmp_path, candidate_pool_per_channel=1, graph_ppr=True,
                     ppr_mix=0.5)
    engine.encode("用户在学习 Java 后端，正在读 Spring 源码")
    engine.encode("用户喜欢自学 Java 原理")
    engine.encode("用户整理了 Java 知识图谱")
    # a cue whose lexical form matches nothing directly still surfaces the
    # java cluster through graph structure
    hits = engine.recall("java 的相关背景梳理", k=5)
    structural = [h for h in hits if h.expanded and
                  any(r.startswith("graph-ppr") for r in h.reasons)]
    assert structural, "PPR should append structurally-central episodes"
    engine.close()


def test_ppr_respects_damping_config(tmp_path):
    engine = _consolidated(tmp_path)
    engine.config = engine.config.model_copy(
        update={"graph_ppr": True, "ppr_damping": 0.5, "ppr_mix": 0.5}
    )
    hits = engine.recall("Java 学习", k=5, touch=False)
    assert hits  # runs with custom damping without error
    engine.close()
