from __future__ import annotations

from brain_memory.episodic.store import content_hash
from brain_memory.extraction.heuristic import HeuristicExperienceParser
from brain_memory.models import MemoryStatus
from brain_memory.storage.db import Database

import numpy as np
import pytest


@pytest.fixture()
def store(tmp_path):
    db = Database(str(tmp_path / "store.db"))
    from brain_memory.episodic.store import EpisodicStore

    yield EpisodicStore(db)
    db.close()


def _embed(text: str, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make(store, text: str, **kwargs):
    extracted = HeuristicExperienceParser().parse(text, **kwargs)
    return store.add(extracted, _embed(text))


def test_add_roundtrip(store):
    episode, duplicate = _make(store, "用户在用 Docker 部署微服务")
    assert duplicate is False
    fetched = store.get(episode.id, with_embedding=True)
    assert fetched.content == episode.content
    assert fetched.embedding.shape == (32,)
    assert fetched.is_active


def test_duplicate_bumps_access_stats(store):
    first, _ = _make(store, "用户在用 Docker 部署微服务")
    second, duplicate = _make(store, "用户在用 Docker 部署微服务")
    assert duplicate is True
    assert second.id == first.id
    assert second.access_count == first.access_count + 1


def test_content_hash_ignores_whitespace():
    assert content_hash("a  b\tc") == content_hash("a b c")
    assert content_hash("a b") != content_hash("a c")


def test_archive_and_restore(store):
    episode, _ = _make(store, "临时记录一条")
    assert store.archive(episode.id) is True
    assert store.get(episode.id).status is MemoryStatus.ARCHIVED
    assert store.restore(episode.id) is True
    assert store.get(episode.id).status is MemoryStatus.ACTIVE


def test_archive_missing_returns_false(store):
    assert store.archive(9999) is False


def test_fts_finds_latin_and_cjk(store):
    _make(store, "用户正在开发 Minecraft P-51 飞机模组")
    _make(store, "用户在研究量子化学")

    latin_hits = store.fts_search('"minecraft"', 5)
    assert len(latin_hits) == 1

    cjk_hits = store.fts_search('"飞机模组"', 5)
    assert len(cjk_hits) == 1


def test_fts_excludes_archived(store):
    episode, _ = _make(store, "已归档的内容 minecraft")
    store.archive(episode.id)
    assert store.fts_search('"minecraft"', 5) == []


def test_entity_tag_index(store):
    episode, _ = _make(store, "用户在学 Java 和 Spring")
    ids = store.ids_for_entities(["java"])
    assert episode.id in ids
    folded = store.ids_for_entities(["JAVA"])
    assert episode.id in folded
