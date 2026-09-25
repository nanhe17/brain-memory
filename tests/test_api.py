from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from brain_memory.api.app import create_app  # noqa: E402
from brain_memory.config import MemoryConfig  # noqa: E402
from brain_memory.engine import MemoryEngine  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    config = MemoryConfig(db_path=str(tmp_path / "api.db"), embedding_dim=64)
    engine = MemoryEngine(config)
    app = create_app(engine)
    with TestClient(app) as test_client:
        yield test_client
    engine.close()


def test_encode_and_recall_endpoints(client):
    response = client.post("/encode", json={"text": "用户在开发 Minecraft P-51 飞机模组"})
    assert response.status_code == 200
    body = response.json()
    assert body["episode"]["id"] == 1
    assert body["duplicate"] is False
    assert "embedding" not in body["episode"]  # vectors never leave the engine

    recall_response = client.post("/recall", json={"cue": "飞机模组", "k": 3})
    assert recall_response.status_code == 200
    hits = recall_response.json()
    assert hits
    assert "score" in hits[0] and "reasons" in hits[0] and "factors" in hits[0]


def test_inspect_and_forget_endpoints(client):
    client.post("/encode", json={"text": "一条可归档的记忆"})
    inspect_response = client.get("/memories/1")
    assert inspect_response.status_code == 200
    assert inspect_response.json()["episode"]["id"] == 1

    forget_response = client.post("/memories/1/forget")
    assert forget_response.status_code == 200
    assert forget_response.json()["status"] == "archived"

    missing = client.get("/memories/999")
    assert missing.status_code == 404


def test_stats_endpoint(client):
    client.post("/encode", json={"text": "统计一下"})
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["total_episodes"] == 1


def test_validation_error(client):
    assert client.post("/encode", json={"text": ""}).status_code == 422
