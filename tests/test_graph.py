from __future__ import annotations

import pytest

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.graph.view import parse_ref
from brain_memory.models import EdgeKind, MemoryStatus, NodeKind


def _engine(tmp_path) -> MemoryEngine:
    return MemoryEngine(MemoryConfig(db_path=str(tmp_path / "graph.db"), embedding_dim=64))


def _consolidated(tmp_path) -> MemoryEngine:
    engine = _engine(tmp_path)
    engine.encode("用户在学习 Java 后端，正在读 Spring 的源码")
    engine.encode("用户喜欢自学 Java 原理")
    engine.encode("用户整理了 Java 知识图谱")
    engine.consolidate()
    return engine


# -- ref parsing ------------------------------------------------------------------


def test_parse_ref_variants():
    assert parse_ref("e12") == (NodeKind.EPISODE, 12, None)
    assert parse_ref(" s3 ") == (NodeKind.SEMANTIC, 3, None)
    assert parse_ref("c:Java") == (NodeKind.CONCEPT, None, "java")
    for bad in ("", "x1", "e", "c:", "eabc"):
        with pytest.raises(ValueError):
            parse_ref(bad)


# -- episode neighborhood -----------------------------------------------------------


def test_episode_neighborhood_mentions_and_explicit(tmp_path):
    engine = _consolidated(tmp_path)
    engine.link("e1", "related_to", "e3")
    sub = engine.neighborhood("e1")

    refs = {n.ref for n in sub.nodes}
    assert "c:java" in refs
    edge_kinds = {(e.kind, e.provenance) for e in sub.edges}
    assert (EdgeKind.MENTIONS, "tags") in edge_kinds
    assert (EdgeKind.RELATED_TO, "explicit") in edge_kinds
    assert sub.center == "e1"
    engine.close()


def test_semantic_neighborhood_evidence_and_conflict(tmp_path):
    engine = _consolidated(tmp_path)
    semantic_id = engine.find_semantic("java").id
    engine.encode("其实我现在不想做 Java 了，转 AI 去了")  # challenge → conflict

    sub = engine.neighborhood(f"s{semantic_id}")
    edge_kinds = {e.kind for e in sub.edges}
    assert EdgeKind.DERIVED_FROM in edge_kinds
    assert EdgeKind.CONTRADICTS in edge_kinds
    contradicts = [e for e in sub.edges if e.kind is EdgeKind.CONTRADICTS]
    assert contradicts[0].provenance == "conflict"
    engine.close()


def test_concept_neighborhood_cooccurrence_and_knowledge(tmp_path):
    engine = _consolidated(tmp_path)
    sub = engine.neighborhood("c:java")

    refs = {n.ref for n in sub.nodes}
    assert "s1" in refs  # the concept's consolidated knowledge
    co_occurs = [e for e in sub.edges if e.kind is EdgeKind.CO_OCCURS_WITH]
    assert co_occurs, "spring co-occurs with java across the episodes"
    assert {t for t, _ in engine.related_concepts("java")} >= {"spring"}
    engine.close()


def test_lifecycle_filters_edges_without_cascade(tmp_path):
    engine = _consolidated(tmp_path)
    semantic_id = engine.find_semantic("java").id
    evidence = engine.find_semantic("java").evidence_ids

    engine.forget(evidence[0])  # FORGOTTEN is the deepest soft state
    sub = engine.neighborhood(f"s{semantic_id}")
    hidden = f"e{evidence[0]}"
    assert hidden not in {n.ref for n in sub.nodes}

    engine.restore(evidence[0])  # soft state: restore brings the edge back
    sub = engine.neighborhood(f"s{semantic_id}")
    assert hidden in {n.ref for n in sub.nodes}
    engine.close()


def test_mermaid_render(tmp_path):
    engine = _consolidated(tmp_path)
    sub = engine.neighborhood("e1")
    text = engine._graph.render_mermaid(sub) if hasattr(engine._graph, "render_mermaid") else None
    from brain_memory.graph.render import render_mermaid

    text = render_mermaid(sub)
    assert text.startswith("graph TD")
    assert "-->|mentions|" in text or "-->|derived_from|" in text or sub.edges == []
    engine.close()


def test_similar_to_edges_on_demand(tmp_path):
    engine = _consolidated(tmp_path)
    sub = engine.neighborhood("e1", include_similar=True)
    similar = [e for e in sub.edges if e.kind is EdgeKind.SIMILAR_TO]
    assert similar
    assert all(0.0 <= e.weight <= 1.0 for e in similar)
    engine.close()


# -- explicit links -------------------------------------------------------------------


def test_link_validation(tmp_path):
    engine = _consolidated(tmp_path)
    with pytest.raises(ValueError):
        engine.link("e1", "loves", "e2")          # outside the closed vocabulary
    with pytest.raises(ValueError):
        engine.link("c:java", "related_to", "e2")  # concepts are not linkable
    with pytest.raises(ValueError):
        engine.link("e1", "related_to", "e1")      # self link
    with pytest.raises(ValueError):
        engine.link("e999", "related_to", "e2")    # missing node
    engine.close()


def test_link_and_unlink_roundtrip(tmp_path):
    engine = _consolidated(tmp_path)
    link = engine.link("e1", "caused_by", "e2", created_by="user")
    assert link.relation == "caused_by"
    sub = engine.neighborhood("e1")
    assert any(
        e.kind.value == "caused_by" and e.provenance == "explicit" for e in sub.edges
    )
    assert engine.unlink(link.id) is True
    sub = engine.neighborhood("e1")
    assert not any(e.kind.value == "caused_by" for e in sub.edges)
    engine.close()


def test_link_to_inactive_endpoint_rejected(tmp_path):
    engine = _consolidated(tmp_path)
    engine.forget(3)
    with pytest.raises(ValueError):
        engine.link("e1", "related_to", "e3")
    engine.close()
