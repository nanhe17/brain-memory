from __future__ import annotations

import numpy as np
import pytest

from brain_memory.retrieval.vector_index import VectorIndex
from brain_memory.storage.db import Database


@pytest.fixture()
def index(tmp_path):
    db = Database(str(tmp_path / "vectors.db"))
    yield VectorIndex(db), db
    db.close()


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _insert(db, index, episode_id: int, vector: np.ndarray) -> None:
    db.insert_episode(
        content=f"episode {episode_id}",
        content_hash=f"hash-{episode_id}",
        entities=[],
        topics=[],
        key_facts=[],
        emphasis=[],
        context=None,
        source="conversation",
        created_at="2026-09-25T00:00:00+00:00",
        embedding=vector.astype(np.float32).tobytes(),
        embedding_dim=int(vector.shape[0]),
        importance=0.5,
        confidence=0.5,
        metadata={},
    )
    index.invalidate()


def test_search_orders_by_similarity(index):
    vindex, db = index
    _insert(db, vindex, 1, _vec(1))
    _insert(db, vindex, 2, _vec(2))
    _insert(db, vindex, 3, _vec(1) * 0.5 + _vec(2) * 0.5)

    hits = vindex.search(_vec(1), k=2)
    assert [episode_id for episode_id, _ in hits][0] == 1
    assert len(hits) == 2
    assert hits[0][1] == pytest.approx(1.0)


def test_search_allowed_ids_filter(index):
    vindex, db = index
    _insert(db, vindex, 1, _vec(1))
    _insert(db, vindex, 2, _vec(2))

    hits = vindex.search(_vec(1), k=5, allowed_ids={2})
    assert [episode_id for episode_id, _ in hits] == [2]
    assert vindex.search(_vec(1), k=5, allowed_ids=set()) == []


def test_search_empty_index(index):
    vindex, _ = index
    assert vindex.search(_vec(1), k=3) == []


def test_dim_mismatch_raises(index):
    vindex, db = index
    _insert(db, vindex, 1, _vec(1, dim=8))
    with pytest.raises(ValueError):
        vindex.search(_vec(1, dim=16), k=1)
