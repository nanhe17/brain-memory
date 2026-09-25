from __future__ import annotations

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine


def test_encode_and_recall_roundtrip(engine):
    result = engine.encode("用户正在开发一个 Minecraft P-51 飞机模组")
    assert result.duplicate is False
    assert result.episode.id == 1
    assert result.episode.entities

    hits = engine.recall("那个飞机模组", k=1)
    assert hits[0].episode.id == result.episode.id


def test_duplicate_encode_bumps_existing(engine):
    first = engine.encode("用户每天用 Docker 构建镜像")
    second = engine.encode("用户每天用 Docker 构建镜像")
    assert second.duplicate is True
    assert second.episode.id == first.episode.id
    assert second.episode.access_count == first.episode.access_count + 1


def test_cross_session_persistence(tmp_path):
    db_path = str(tmp_path / "persist.db")
    config = MemoryConfig(db_path=db_path, embedding_dim=64)

    with MemoryEngine(config) as engine:
        engine.encode("用户的项目代号是 nightingale，一个 Minecraft 模组项目")

    with MemoryEngine(config) as engine:
        hits = engine.recall("飞机模组项目代号", k=3)
        assert hits
        assert any("nightingale" in h.episode.content for h in hits)


def test_forget_is_soft_and_reversible(engine):
    episode = engine.encode("敏感的临时草稿").episode
    assert engine.forget(episode.id) is True
    assert engine.recall("敏感草稿", k=5) == []
    inspection = engine.inspect(episode.id)
    assert inspection is not None
    assert inspection["episode"].status.value == "archived"
    assert engine.restore(episode.id) is True
    assert engine.recall("敏感草稿", k=5)


def test_similar_but_different_experiences_stay_separate(engine):
    """Pattern separation at the episodic layer: no merging, ever."""
    engine.encode("用户喜欢 Java")
    engine.encode("用户其实不喜欢 Java 了")
    engine.encode("用户重新开始喜欢 Java")
    assert engine.stats().active == 3

    hits = engine.recall("Java 喜好", k=5)
    contents = [h.episode.content for h in hits]
    assert len([c for c in contents if "Java" in c]) == 3
    ids = [h.episode.id for h in hits]
    assert len(ids) == len(set(ids))


def test_inspect_includes_related(engine):
    engine.encode("用户正在开发 Minecraft 飞机模组")
    engine.encode("模组的渲染部分用的是 Minecraft Forge 管线")
    engine.encode("完全无关：今天的午饭是面条")
    inspection = engine.inspect(1)
    assert inspection["episode"].id == 1
    related_ids = [e.id for e in inspection["related"]]
    assert 2 in related_ids


def test_inspect_missing_returns_none(engine):
    assert engine.inspect(9999) is None


def test_stats_shape(engine):
    engine.encode("一条记忆")
    engine.encode("另一条记忆")
    engine.forget(1)
    stats = engine.stats()
    assert stats.total_episodes == 2
    assert stats.active == 1
    assert stats.archived == 1
    assert 0.0 <= stats.avg_importance <= 1.0


def test_empty_recall(engine):
    assert engine.recall("完全陌生的话题", k=3) == []
