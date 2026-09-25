from __future__ import annotations

import httpx
import pytest
from datetime import datetime, timedelta, timezone

from brain_memory.consolidation.llm import LLMConsolidator
from brain_memory.models import Episode, SemanticKind


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _episodes(n: int = 4) -> list[Episode]:
    return [
        Episode(
            id=i,
            content=f"Java learning session {i}",
            content_hash=f"h{i}",
            entities=["Java"],
            topics=["java"],
            created_at=datetime.now(timezone.utc) - timedelta(days=i),
            importance=0.5,
            confidence=0.6,
        )
        for i in range(1, n + 1)
    ]


def _consolidator() -> LLMConsolidator:
    return LLMConsolidator(base_url="http://fake/v4", api_key="k", model="glm-4")


def test_proposal_parsed_with_support_gate(monkeypatch):
    payload = (
        '{"statement": "User is systematically learning the Java backend stack.", '
        '"kind": "generalization", "confidence": 0.85, "supporting": [0, 1, 2]}'
    )
    monkeypatch.setattr(
        "brain_memory.consolidation.llm.httpx.post", lambda *a, **kw: _FakeResponse(payload)
    )
    proposal = _consolidator().propose("java", _episodes(4), min_support=3)
    assert proposal is not None
    assert proposal.kind is SemanticKind.GENERALIZATION
    assert proposal.supporting_indexes == [0, 1, 2]
    assert proposal.confidence == pytest.approx(0.85)


def test_over_generalization_fails_quality_gate(monkeypatch):
    # LLM claims a pattern but only 1 episode supports it -> None
    payload = '{"statement": "User loves everything.", "kind": "preference", "confidence": 0.9, "supporting": [0]}'
    monkeypatch.setattr(
        "brain_memory.consolidation.llm.httpx.post", lambda *a, **kw: _FakeResponse(payload)
    )
    assert _consolidator().propose("java", _episodes(4), min_support=3) is None


def test_no_pattern_found_returns_none(monkeypatch):
    payload = '{"statement": "", "supporting": []}'
    monkeypatch.setattr(
        "brain_memory.consolidation.llm.httpx.post", lambda *a, **kw: _FakeResponse(payload)
    )
    assert _consolidator().propose("java", _episodes(4), min_support=3) is None


def test_invalid_kind_falls_back_to_fact(monkeypatch):
    payload = '{"statement": "Something stable.", "kind": "vibes", "confidence": 0.7, "supporting": [0, 1, 2]}'
    monkeypatch.setattr(
        "brain_memory.consolidation.llm.httpx.post", lambda *a, **kw: _FakeResponse(payload)
    )
    proposal = _consolidator().propose("java", _episodes(3), min_support=3)
    assert proposal.kind is SemanticKind.FACT


def test_transport_failure_raises(monkeypatch):
    def boom(*a, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("brain_memory.consolidation.llm.httpx.post", boom)
    with pytest.raises(httpx.ConnectError):
        _consolidator().propose("java", _episodes(3), min_support=3)


def test_requires_key():
    with pytest.raises(ValueError):
        LLMConsolidator(base_url="http://fake", api_key="", model="m")
