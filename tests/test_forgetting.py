from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.forgetting.decay import (
    episode_strength,
    last_touched,
    semantic_strength,
)
from brain_memory.models import Episode, MemoryStatus, SemanticKind, SemanticMemory

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
HALF_LIFE = 14.0


def _episode(days_ago: float, *, importance=0.30, confidence=0.55, access_count=0,
             episode_id: int = 1, content: str = "chit chat") -> Episode:
    return Episode(
        id=episode_id,
        content=content,
        content_hash=f"h{episode_id}",
        created_at=NOW - timedelta(days=days_ago),
        importance=importance,
        confidence=confidence,
        access_count=access_count,
    )


def _semantic(evidence: int, *, days_ago: float = 100.0, confidence=0.5,
              semantic_id: int = 1) -> SemanticMemory:
    return SemanticMemory(
        id=semantic_id,
        concept="java",
        statement="java knowledge",
        kind=SemanticKind.CO_OCCURRENCE,
        confidence=confidence,
        evidence_ids=list(range(1, evidence + 1)),
        created_at=NOW - timedelta(days=days_ago + 1),
        updated_at=NOW - timedelta(days=days_ago),
    )


# -- strength math ----------------------------------------------------------------


def test_last_touched_prefers_access():
    created = NOW - timedelta(days=100)
    accessed = NOW - timedelta(days=1)
    assert last_touched(created, accessed) == accessed
    assert last_touched(created, None) == created


def test_fresh_memory_is_strong():
    assert episode_strength(_episode(0), now=NOW, half_life_days=HALF_LIFE) > 0.5


def test_stale_chit_chat_decays_below_archive_threshold():
    # 0.30*0.30 + ~0 recency + 0.20*0.55 ≈ 0.20 < 0.25
    strength = episode_strength(_episode(90), now=NOW, half_life_days=HALF_LIFE)
    assert strength < 0.25


def test_recent_access_keeps_old_memories_strong():
    episode = _episode(200, access_count=10)
    episode.last_accessed = NOW - timedelta(days=1)
    assert episode_strength(episode, now=NOW, half_life_days=HALF_LIFE) > 0.45


def test_important_memories_survive_200_days():
    # "请记住"-style memory: importance 0.87, confidence 0.9
    strength = episode_strength(
        _episode(200, importance=0.87, confidence=0.90),
        now=NOW, half_life_days=HALF_LIFE,
    )
    assert strength > 0.25


def test_semantic_strength_is_evidence_weighted():
    thin = semantic_strength(_semantic(1), now=NOW, half_life_days=HALF_LIFE)
    rich = semantic_strength(_semantic(5), now=NOW, half_life_days=HALF_LIFE)
    assert rich > thin
    assert rich > 0.4  # full evidence support dominates even with stale recency
    assert thin < 0.25  # thin evidence alone cannot hold the archive threshold


def test_semantic_strength_decays_with_time():
    assert semantic_strength(_semantic(1), now=NOW, half_life_days=HALF_LIFE) > \
        semantic_strength(_semantic(1), now=NOW + timedelta(days=200),
                          half_life_days=HALF_LIFE)


# -- sweep behavior -----------------------------------------------------------------


def _engine(tmp_path, **overrides) -> MemoryEngine:
    return MemoryEngine(MemoryConfig(db_path=str(tmp_path / "decay.db"),
                                     embedding_dim=64, **overrides))


def _seed_mixed(engine: MemoryEngine) -> None:
    engine.encode("用户在学习 Java 后端，正在读 Spring 的源码",
                  created_at=NOW - timedelta(days=40))
    engine.encode("用户喜欢自学 Java 原理", created_at=NOW - timedelta(days=35))
    engine.encode("用户整理了 Java 知识图谱", created_at=NOW - timedelta(days=30))
    engine.encode("今天聊了聊天气真不错", created_at=NOW - timedelta(days=90))
    engine.encode("用户抱怨地铁太挤了", created_at=NOW - timedelta(days=85))


def test_sweep_archives_stale_chit_chat_and_keeps_knowledge(tmp_path):
    engine = _engine(tmp_path)
    _seed_mixed(engine)
    engine.consolidate()

    report = engine.decay()
    assert set(report.archived_episode_ids) == {4, 5}
    assert report.archived_semantic_ids == []  # fresh knowledge survives
    assert report.protected_evidence_count == 3  # java episodes are evidence

    # evidence episodes untouched, junk gone from recall
    assert all(engine._store.get(i).status is MemoryStatus.ACTIVE for i in (1, 2, 3))
    assert engine._store.get(4).status is MemoryStatus.ARCHIVED
    hits = engine.recall("Java 学习", k=5)
    assert any(h.is_semantic for h in hits)
    engine.close()


def test_dry_run_reports_without_transitioning(tmp_path):
    engine = _engine(tmp_path)
    _seed_mixed(engine)
    engine.consolidate()
    report = engine.decay(dry_run=True)
    assert report.dry_run is True
    assert set(report.archived_episode_ids) == {4, 5}
    assert engine._store.get(4).status is MemoryStatus.ACTIVE  # nothing happened
    assert engine.stats().active == 5
    engine.close()


def test_decay_is_idempotent(tmp_path):
    engine = _engine(tmp_path)
    _seed_mixed(engine)
    engine.consolidate()
    first = engine.decay(now=NOW)   # explicit now: dwell arithmetic must not
    second = engine.decay(now=NOW)  # depend on wall-clock drift in tests
    assert first.transitions > 0
    assert second.transitions == 0
    engine.close()


def test_dwell_time_drives_forgetting(tmp_path):
    engine = _engine(tmp_path)
    _seed_mixed(engine)
    engine.decay()  # junk -> archived, last_touched ~90d ago

    # 100 days later: archived episodes crossed the dwell threshold
    report = engine.decay(now=NOW + timedelta(days=100))
    assert set(report.forgotten_episode_ids) == {4, 5}
    assert engine._store.get(4).status is MemoryStatus.FORGOTTEN

    # forgotten is soft: restore brings it back
    assert engine.restore(4) is True
    assert engine._store.get(4).status is MemoryStatus.ACTIVE
    engine.close()


def test_semantic_with_thin_evidence_decays_over_time(tmp_path):
    engine = _engine(tmp_path, consolidation_min_support=2)
    engine.encode("用户在学习 Java 后端")
    engine.encode("用户喜欢自学 Java 原理")
    engine.consolidate()
    semantic_id = engine.find_semantic("java").id
    assert semantic_id is not None

    # two-evidence knowledge: support bonus 0.12; after ~200d recency is gone
    # 0.25*~0.36 + 0.30*0.4 ≈ 0.21 < 0.25 -> archived
    report = engine.decay(now=NOW + timedelta(days=200))
    assert semantic_id in report.archived_semantic_ids
    assert engine.find_semantic("java").status is MemoryStatus.ARCHIVED

    # released evidence now ages: nothing protects it anymore.  Sequential
    # transitions: an episode archived in one sweep can only be forgotten by
    # a LATER sweep (its dwell clock started at creation).
    engine.decay(now=NOW + timedelta(days=300))
    assert engine._store.get(1).status is MemoryStatus.ARCHIVED
    final = engine.decay(now=NOW + timedelta(days=400))
    assert 1 in final.forgotten_episode_ids
    engine.close()


def test_thresholds_are_configurable(tmp_path):
    engine = _engine(tmp_path, decay_archive_threshold=0.9)  # everything archives
    _seed_mixed(engine)
    engine.consolidate()
    report = engine.decay()
    assert set(report.archived_episode_ids) == {4, 5}  # java episodes protected
    assert report.archived_semantic_ids == [1]  # knowledge itself swept
    engine.close()


def test_archived_rows_excluded_from_consolidation_groups(tmp_path):
    """Regression: archived/forgotten rows must not feed tag_group_counts —
    otherwise decayed episodes would keep consolidating into knowledge."""
    engine = _engine(tmp_path)
    for days, tag in ((200, "后端"), (190, "原理"), (180, "图谱")):
        engine.encode(f"用户在学习 Java 的{tag}", created_at=NOW - timedelta(days=days))
    report = engine.decay()
    assert set(report.archived_episode_ids) == {1, 2, 3}

    result = engine.consolidate()
    # exclusion is upstream: with zero active episodes the java group does
    # not exist at all — no knowledge is created from decayed rows
    assert not result.created
    assert engine.find_semantic("java") is None
    engine.close()
