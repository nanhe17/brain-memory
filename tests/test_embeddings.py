from __future__ import annotations

import numpy as np
import pytest

from brain_memory.embeddings.hash_embedder import HashEmbedder
from brain_memory.embeddings.openai_compatible import OpenAICompatibleEmbedder


def test_deterministic_and_normalized():
    embedder = HashEmbedder(dim=64)
    a = embedder.embed_texts(["hello world", "hello world"])
    assert a.shape == (2, 64)
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a[0]), 1.0)


def test_related_texts_share_signal_more_than_unrelated():
    embedder = HashEmbedder(dim=64)
    vectors = embedder.embed_texts(
        ["minecraft forge mod", "minecraft forge modding", "quantum chemistry bonds"]
    )
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated
    assert related > 0.0


def test_empty_text_yields_zero_vector():
    embedder = HashEmbedder(dim=64)
    vectors = embedder.embed_texts([""])
    assert np.allclose(vectors[0], 0.0)


def test_dim_too_small_rejected():
    with pytest.raises(ValueError):
        HashEmbedder(dim=8)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_openai_compatible_parses_and_normalizes(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        index = {text: i for i, text in enumerate(json["input"])}
        return _FakeResponse(
            {
                "data": [
                    {"index": index[text], "embedding": [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]}
                    for i, text in enumerate(json["input"])
                ]
            }
        )

    monkeypatch.setattr("brain_memory.embeddings.openai_compatible.httpx.post", fake_post)
    embedder = OpenAICompatibleEmbedder(
        base_url="http://fake/v4", api_key="k", model="embedding-3", retries=0
    )
    vectors = embedder.embed_texts(["a", "b"])
    assert embedder.dim == 2
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert len(calls) == 1  # batched into a single request


def test_openai_compatible_retries_then_raises(monkeypatch):
    import httpx

    attempts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        attempts.append(1)
        raise httpx.ConnectError("transport down")

    monkeypatch.setattr("brain_memory.embeddings.openai_compatible.httpx.post", fake_post)
    monkeypatch.setattr("brain_memory.embeddings.openai_compatible.time.sleep", lambda s: None)
    embedder = OpenAICompatibleEmbedder(
        base_url="http://fake/v4", api_key="k", model="m", retries=2
    )
    with pytest.raises(Exception):
        embedder.embed_texts(["x"])
    assert len(attempts) == 3


def test_openai_compatible_requires_key():
    with pytest.raises(ValueError):
        OpenAICompatibleEmbedder(base_url="http://fake", api_key="", model="m")
