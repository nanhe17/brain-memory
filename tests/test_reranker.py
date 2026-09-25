from __future__ import annotations

import httpx
import pytest

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.models import Episode, FactorScores, RecallResult
from brain_memory.retrieval.reranker import LLMReranker
from datetime import datetime, timezone


def _episode(episode_id: int, content: str) -> RecallResult:
    return RecallResult(
        episode=Episode(
            id=episode_id,
            content=content,
            content_hash=f"h{episode_id}",
            created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            importance=0.5,
            confidence=0.5,
        ),
        score=0.5,
        factors=FactorScores(),
    )


# -- score parsing ----------------------------------------------------------------


def test_parse_scores_clamps_and_neutralizes():
    raw = '[{"id": 0, "relevance": 1.7}, {"id": 2, "relevance": -0.3}]'
    scores = LLMReranker._parse_scores(raw, count=3)
    assert scores == [1.0, 0.5, 0.0]  # clamped; missing id 1 stays neutral


def test_parse_scores_ignores_garbage_entries():
    raw = '[{"id": 9, "relevance": 1.0}, {"id": 1, "relevance": "high"}, {"id": 0, "relevance": 0.8}, "junk"]'
    scores = LLMReranker._parse_scores(raw, count=2)
    assert scores == [0.8, 0.5]


def test_parse_scores_code_fenced():
    raw = '```json\n[{"id": 0, "relevance": 0.9}]\n```'
    assert LLMReranker._parse_scores(raw, count=1) == [0.9]


def test_parse_scores_rejects_non_array():
    with pytest.raises(ValueError):
        LLMReranker._parse_scores('{"id": 0}', count=1)


# -- transport ---------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_rerank_call_and_alignment(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["json"] = json
        return _FakeResponse('[{"id": 0, "relevance": 0.2}, {"id": 1, "relevance": 0.9}]')

    monkeypatch.setattr("brain_memory.retrieval.reranker.httpx.post", fake_post)
    reranker = LLMReranker(base_url="http://fake/v4", api_key="k", model="glm-4")
    candidates = [_episode(11, "alpha"), _episode(22, "beta")]
    scores = reranker.rerank("query", candidates)
    assert scores == [0.2, 0.9]
    assert len(seen["json"]["messages"]) == 2


def test_rerank_retries_then_raises(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        raise httpx.ConnectError("down")

    monkeypatch.setattr("brain_memory.retrieval.reranker.httpx.post", fake_post)
    reranker = LLMReranker(base_url="http://fake", api_key="k", model="m", retries=2)
    with pytest.raises(RuntimeError):
        reranker.rerank("query", [_episode(1, "x")])
    assert len(calls) == 3


def test_reranker_requires_key():
    with pytest.raises(ValueError):
        LLMReranker(base_url="http://fake", api_key="", model="m")


# -- fusion in the engine ------------------------------------------------------------


class _FixedReranker:
    name = "fixed"

    def __init__(self, scores):
        self._scores = scores

    def rerank(self, cue, candidates):
        return self._scores


class _ExplodingReranker:
    name = "boom"

    def rerank(self, cue, candidates):
        raise RuntimeError("no llm today")


def _engine(tmp_path, **config_overrides) -> MemoryEngine:
    config = MemoryConfig(
        db_path=str(tmp_path / "r.db"), embedding_dim=64, **config_overrides
    )
    return MemoryEngine(config)


def test_fusion_reorders_and_annotates(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户喜欢 Python")            # id 1: strong lexical/semantic match
    engine.encode("用户在研究量子化学")          # id 2: weak match
    baseline = engine.recall("Python 喜好", k=2, touch=False)
    assert baseline[0].episode.id == 1

    # blind judge loves episode 2 despite weak factors; mix=0.9 flips the order
    engine._reranker = _FixedReranker([0.05, 1.0])
    engine.config = engine.config.model_copy(update={"rerank_mix": 0.9})
    fused = engine.recall("Python 喜好", k=2, touch=False)
    assert fused[0].episode.id == 2
    assert fused[0].llm_relevance == pytest.approx(1.0)
    assert any("llm_relevance" in reason for reason in fused[0].reasons)
    # fusion math: 0.9*1.0 + 0.1*factor_score
    expected = 0.9 * 1.0 + 0.1 * baseline[1].score
    assert fused[0].score == pytest.approx(expected, abs=1e-9)
    engine.close()


def test_fusion_failure_keeps_factor_ranking(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户喜欢 Python")
    engine.encode("用户在研究量子化学")
    baseline = engine.recall("Python 喜好", k=2, touch=False)

    engine._reranker = _ExplodingReranker()
    fused = engine.recall("Python 喜好", k=2, touch=False)
    assert [r.episode.id for r in fused] == [r.episode.id for r in baseline]
    assert all(r.llm_relevance is None for r in fused)
    engine.close()


def test_rerank_off_by_config(tmp_path):
    engine = _engine(
        tmp_path, llm_model="glm-4", llm_api_key="k", rerank="off"
    )
    assert engine._reranker is None
    engine.close()


def test_rerank_single_result_skipped(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("唯一的一条记忆")
    engine._reranker = _FixedReranker([0.0])
    hits = engine.recall("唯一记忆", k=1, touch=False)
    assert hits[0].llm_relevance is None  # one candidate: nothing to rerank
    engine.close()
