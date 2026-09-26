from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from brain_memory.api.app import create_app  # noqa: E402
from brain_memory.api.svg import subgraph_to_svg  # noqa: E402
from brain_memory.config import MemoryConfig  # noqa: E402
from brain_memory.engine import MemoryEngine  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    config = MemoryConfig(db_path=str(tmp_path / "inspector.db"), embedding_dim=64)
    engine = MemoryEngine(config)
    engine.encode("用户在学习 Java 后端，正在读 Spring 源码")
    engine.encode("用户喜欢自学 Java 原理")
    engine.encode("用户整理了 Java 知识图谱")
    engine.consolidate()
    app = create_app(engine)
    with TestClient(app) as test_client:
        yield test_client, engine
    engine.close()


def test_inspector_html_served(client):
    test_client, _engine = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Brain Memory Inspector" in response.text


def test_overview_endpoint(client):
    test_client, engine = client
    body = test_client.get("/api/overview").json()
    assert body["stats"]["semantic_memories"] == 1
    assert body["embedding_provider"].startswith("hash")
    assert "archive_threshold" in body["decay"]


def test_timeline_endpoint(client):
    test_client, _engine = client
    body = test_client.get("/api/timeline?days=30").json()
    assert body["series"]
    today = body["series"][-1]
    assert today["episodes"] >= 3
    assert today["semantics"] >= 1


def test_memories_endpoint_filters(client):
    test_client, engine = client
    body = test_client.get("/api/memories?status=active&limit=2").json()
    assert len(body["episodes"]) <= 2
    assert body["total"] == 3
    assert body["semantics"][0]["concept"] == "java"

    engine.forget(1)
    archived = test_client.get("/api/memories?status=archived").json()
    assert any(e["id"] == 1 for e in archived["episodes"])
    engine.restore(1)


def test_detail_endpoints(client):
    test_client, _engine = client
    episode = test_client.get("/api/memories/episode/1")
    assert episode.status_code == 200
    assert episode.json()["episode"]["id"] == 1

    semantic = test_client.get("/api/memories/semantic/1")
    assert semantic.status_code == 200
    body = semantic.json()
    assert len(body["evidence"]) == 3
    assert body["versions"][0]["version"] == 1

    assert test_client.get("/api/memories/episode/999").status_code == 404
    assert test_client.get("/api/memories/semantic/999").status_code == 404


def test_graph_endpoints(client):
    test_client, _engine = client
    svg = test_client.get("/api/graph/e1.svg")
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in svg.text

    data = test_client.get("/api/graph/c:java.json")
    assert data.status_code == 200
    assert any(e["kind"] == "co_occurs_with" for e in data.json()["edges"])

    missing = test_client.get("/api/graph/e999.svg")
    assert missing.status_code == 404


def test_strengths_conflicts_decay_preview(client):
    test_client, engine = client
    strengths = test_client.get("/api/strengths?limit=5").json()
    assert len(strengths) == 4  # 3 episodes + 1 semantic
    values = [s["strength"] for s in strengths]
    assert values == sorted(values)

    assert test_client.get("/api/conflicts").json() == []

    engine.encode("其实我现在不想做 Java 了，转 AI 去了")
    conflicts = test_client.get("/api/conflicts?status=open").json()
    assert len(conflicts) == 1
    assert test_client.get("/api/decay/preview").json()["dry_run"] is True


def test_svg_renderer_deterministic_and_escapes():
    from brain_memory.graph.view import GraphView  # noqa: F401 — sanity import

    # direct unit check with a minimal subgraph-shaped object
    from brain_memory.models import (
        EdgeKind,
        GraphEdge,
        GraphNode,
        GraphSubgraph,
        NodeKind,
    )

    sub = GraphSubgraph(
        center="e1",
        nodes=[
            GraphNode(ref="e1", kind=NodeKind.EPISODE, id=1, label='bad "quote"'),
            GraphNode(ref="c:java", kind=NodeKind.CONCEPT, label="java"),
        ],
        edges=[GraphEdge(source="e1", target="c:java", kind=EdgeKind.MENTIONS,
                         provenance="tags")],
    )
    first = subgraph_to_svg(sub)
    second = subgraph_to_svg(sub)
    assert first == second
    assert "<svg" in first and 'bad "quote"' not in first
