from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from brain_memory.extraction.heuristic import HeuristicExperienceParser
from brain_memory.retrieval.retriever import Retriever, build_fts_match_expr


@pytest.fixture()
def populated(engine):
    now = datetime.now(timezone.utc)
    engine.encode("用户正在开发一个 Minecraft P-51 飞机模组，用 Forge 做三视图建模",
                  created_at=now - timedelta(days=1))
    engine.encode("用户在学习 Java 后端和 Spring 微服务",
                  created_at=now - timedelta(days=2))
    engine.encode("今天天气很好，出去散步了",
                  created_at=now - timedelta(days=3))
    return engine


def test_vector_and_keyword_channels_find_relevant(populated):
    hits = populated.recall("继续昨天那个飞机模组", k=3)
    assert hits, "expected at least one hit"
    assert "飞机" in hits[0].episode.content or "Minecraft" in hits[0].episode.content
    assert hits[0].score > 0
    assert hits[0].reasons, "explanations must be present"
    for factor_value in hits[0].factors.as_dict().values():
        assert 0.0 <= factor_value <= 1.0


def test_keyword_channel_hit_via_rare_token(populated):
    hits = populated.recall("Forge 三视图", k=3)
    top_contents = [h.episode.content for h in hits]
    assert any("Forge" in content for content in top_contents)


def test_archived_memories_are_not_recalled(populated):
    relevant = populated.recall("飞机模组", k=1)[0]
    populated.forget(relevant.episode.id)
    hits = populated.recall("飞机模组", k=3)
    assert all(h.episode.id != relevant.episode.id for h in hits)


def test_source_filter(populated):
    engine = populated
    engine.encode("用户提到了 Kafka 消息队列", source="note")
    hits = engine.recall("Kafka", k=5, source="conversation")
    assert all(h.episode.source == "conversation" for h in hits)


def test_require_entities_filter(populated):
    hits = populated.recall("Java Spring 学习进展", k=5, require_entities=["Java"])
    assert hits
    assert all("Java" in h.episode.entities for h in hits)


def test_time_range_filter(populated):
    now = datetime.now(timezone.utc)
    hits = populated.recall("飞机模组", k=5, time_from=now - timedelta(days=1, hours=12))
    assert all(h.episode.created_at >= now - timedelta(days=1, hours=12) for h in hits)


def test_recall_touches_access_stats(populated):
    before = populated.recall("飞机模组", k=1)[0].episode.access_count
    populated.recall("飞机模组", k=1)
    after = populated.recall("飞机模组", k=1, touch=False)[0].episode.access_count
    assert after >= before + 1


def test_fts_match_expr_shape():
    cue = HeuristicExperienceParser().parse('继续 "P-51" 那个飞机模组 forge')
    expr = build_fts_match_expr(cue)
    assert expr, "expression must not be empty"
    assert expr.count(" OR ") >= 1
    for term in expr.split(" OR "):
        assert term.startswith('"') and term.endswith('"')


def test_retriever_direct_use(populated):
    retriever = Retriever(
        populated._store, populated._index, populated.config
    )
    cue = HeuristicExperienceParser().parse("飞机模组")
    vector = populated.embedder.embed_texts(["飞机模组"])[0]
    results = retriever.retrieve([cue], [vector], k=2)
    assert 1 <= len(results) <= 2


def test_retriever_multi_variant_union(populated):
    retriever = Retriever(
        populated._store, populated._index, populated.config
    )
    parser = HeuristicExperienceParser()
    cue_a = parser.parse("飞机模组")
    cue_b = parser.parse("Minecraft Forge 三视图")
    texts = [cue_a.content, cue_b.content]
    vectors = populated.embedder.embed_texts(texts)
    results = retriever.retrieve([cue_a, cue_b], list(vectors), k=5)
    assert results
    # a variant that names 三视图 must lift the Forge episode via keyword channel
    assert any("Forge" in r.episode.content for r in results)
    # factor sets union across variants
    assert results[0].factors.keyword > 0 or results[0].factors.semantic > 0
