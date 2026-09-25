from __future__ import annotations

import httpx
import pytest

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.query.expander import LLMQueryExpander, QueryExpansion


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _expander() -> LLMQueryExpander:
    return LLMQueryExpander(base_url="http://fake/v4", api_key="k", model="glm-4")


def test_expand_parses_llm_json(monkeypatch):
    payload = (
        '```json\n{"rewritten": "P-51 Mustang 飞机模组 进展", '
        '"sub_queries": ["P-51 模组"], "entities": ["P-51"], '
        '"time_from": "2026-09-20T00:00:00+00:00", "time_to": null}\n```'
    )
    monkeypatch.setattr(
        "brain_memory.query.expander.httpx.post", lambda *a, **kw: _FakeResponse(payload)
    )
    expansion = _expander().expand("那个模组怎么样了", "goal: continue mod")
    assert expansion is not None
    assert expansion.rewritten == "P-51 Mustang 飞机模组 进展"
    assert expansion.sub_queries == ["P-51 模组"]
    assert expansion.time_from is not None


def test_expand_returns_none_on_transport_error(monkeypatch):
    def boom(*a, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("brain_memory.query.expander.httpx.post", boom)
    assert _expander().expand("cue", "") is None


def test_expand_returns_none_on_malformed_json(monkeypatch):
    monkeypatch.setattr(
        "brain_memory.query.expander.httpx.post",
        lambda *a, **kw: _FakeResponse("not json at all"),
    )
    assert _expander().expand("cue", "") is None


def test_expand_returns_none_when_rewritten_empty(monkeypatch):
    monkeypatch.setattr(
        "brain_memory.query.expander.httpx.post",
        lambda *a, **kw: _FakeResponse('{"rewritten": "", "sub_queries": []}'),
    )
    assert _expander().expand("cue", "") is None


def test_expander_requires_key():
    with pytest.raises(ValueError):
        LLMQueryExpander(base_url="http://fake", api_key="", model="m")


# -- engine integration ---------------------------------------------------------


class _FakeExpander:
    def __init__(self, expansion: QueryExpansion | None):
        self._expansion = expansion

    def expand(self, cue: str, working_context: str) -> QueryExpansion | None:
        return self._expansion


def _engine(tmp_path, llm: bool) -> MemoryEngine:
    config = MemoryConfig(
        db_path=str(tmp_path / "e.db"),
        embedding_dim=64,
        llm_model="glm-4" if llm else "",
        llm_api_key="k" if llm else "",
    )
    return MemoryEngine(config)


def test_expansion_variant_lifts_lexically_disjoint_cue(tmp_path):
    engine = _engine(tmp_path, llm=True)
    engine.encode("用户在做 P-51 Mustang 飞机的三视图建模")   # id 1
    engine.encode("用户在超市买了牛奶和面包")                  # id 2
    engine.encode("用户在上英语课")                           # id 3

    # no expansion: cue shares no tokens with episode 1 -> miss
    baseline = engine.recall("那个战斗机项目怎么样了", k=1, use_working_memory=False)
    assert baseline[0].episode.id != 1

    engine._expander = _FakeExpander(
        QueryExpansion(rewritten="P-51 Mustang 三视图建模 进展")
    )
    hits = engine.recall("那个战斗机项目怎么样了", k=1, use_working_memory=False)
    assert hits[0].episode.id == 1
    engine.close()


def test_expansion_off_by_config(tmp_path):
    config = MemoryConfig(
        db_path=str(tmp_path / "e.db"),
        embedding_dim=64,
        llm_model="glm-4",
        llm_api_key="k",
        query_expansion="off",
    )
    engine = MemoryEngine(config)
    assert engine._expander is None
    engine.close()


def test_expansion_auto_requires_llm(tmp_path):
    engine = _engine(tmp_path, llm=False)
    assert engine._expander is None
    engine.close()


def test_expansion_failure_degrades_to_original(tmp_path):
    engine = _engine(tmp_path, llm=True)
    engine.encode("用户在写 Rust 异步服务")

    class _Boom:
        def expand(self, cue, ctx):
            raise RuntimeError("llm exploded")

    engine._expander = _Boom()
    hits = engine.recall("Rust 异步服务", k=1)
    assert hits and hits[0].episode.id == 1
    engine.close()


def test_expansion_supplies_time_filter(tmp_path):
    from datetime import datetime, timedelta, timezone

    engine = _engine(tmp_path, llm=True)
    old = engine.encode("用户完成了 K8s 集群升级", created_at=datetime.now(timezone.utc) - timedelta(days=30))
    engine.encode("用户今天在学 Rust")

    engine._expander = _FakeExpander(
        QueryExpansion(
            rewritten="K8s 集群升级",
            time_from=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        )
    )
    hits = engine.recall("集群升级", k=5)
    assert all(h.episode.id != old.episode.id for h in hits)
    engine.close()


def test_parse_iso_handles_naive_and_invalid():
    from datetime import timezone

    assert MemoryEngine._parse_iso(None) is None
    assert MemoryEngine._parse_iso("garbage") is None
    parsed = MemoryEngine._parse_iso("2026-09-01T00:00:00")
    assert parsed is not None and parsed.tzinfo is timezone.utc
