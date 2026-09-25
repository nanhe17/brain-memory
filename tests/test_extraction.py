from __future__ import annotations

import pytest

from brain_memory.extraction.heuristic import HeuristicExperienceParser


@pytest.fixture()
def parser():
    return HeuristicExperienceParser()


def test_entities_from_quotes_camelcase_and_lexicon(parser):
    extracted = parser.parse('我们在 "P-51 Mustang" 项目里用了 SpringBoot 和 Minecraft Forge')
    entities = [e.casefold() for e in extracted.entities]
    assert "p-51 mustang" in entities
    assert "springboot" in entities


def test_entities_include_cjk_lexicon(parser):
    extracted = parser.parse("这个飞机模组需要新的渲染管线")
    assert "飞机" in extracted.entities
    assert "模组" in extracted.entities


def test_topics_are_lexicon_canonical(parser):
    extracted = parser.parse("I am learning Java and Spring for backend work")
    assert "java" in extracted.topics
    assert "spring" in extracted.topics


def test_explicit_remember_raises_importance(parser):
    plain = parser.parse("用户在写一个爬虫")
    explicit = parser.parse("请记住，用户在写一个爬虫")
    assert explicit.importance > plain.importance
    assert "explicit_remember" in explicit.emphasis_signals


def test_question_scores_lower_than_statement(parser):
    question = parser.parse("Python 的 GIL 是什么？")
    statement = parser.parse("用户每天用 Python 写爬虫")
    assert question.importance < statement.importance


def test_determinism(parser):
    from datetime import datetime, timezone

    text = "用户喜欢用 Docker 部署微服务，重要的是保持简单"
    when = datetime(2026, 9, 25, tzinfo=timezone.utc)
    first = parser.parse(text, timestamp=when)
    second = parser.parse(text, timestamp=when)
    assert first == second


def test_empty_text_rejected(parser):
    with pytest.raises(ValueError):
        parser.parse("   ")


def test_timestamp_override(parser):
    from datetime import datetime, timezone

    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    extracted = parser.parse("发生了某事", timestamp=when)
    assert extracted.timestamp == when


def test_confidence_within_bounds(parser):
    extracted = parser.parse("随便说说而已")
    assert 0.0 <= extracted.importance <= 1.0
    assert 0.0 <= extracted.confidence <= 1.0
