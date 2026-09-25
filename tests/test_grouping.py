from __future__ import annotations

import pytest

from brain_memory.consolidation.grouping import candidate_groups
from brain_memory.storage.db import Database


@pytest.fixture()
def db(tmp_path):
    database = Database(str(tmp_path / "groups.db"))
    yield database
    database.close()


def _episode(db, episode_id: int, entities: list[str], topics: list[str], content: str = "content") -> None:
    db.insert_episode(
        content=f"{content} {episode_id}",
        content_hash=f"hash-{episode_id}",
        entities=entities,
        topics=topics,
        key_facts=[],
        emphasis=[],
        context=None,
        source="conversation",
        created_at="2026-09-01T00:00:00+00:00",
        embedding=None,
        embedding_dim=0,
        importance=0.5,
        confidence=0.5,
        metadata={},
    )
    tags = [(episode_id, "entity", e) for e in entities] + [(episode_id, "topic", t) for t in topics]
    db.insert_tags(tags)


def test_min_support_filters_small_groups(db):
    _episode(db, 1, ["java"], [])
    _episode(db, 2, ["java"], [])
    groups = candidate_groups(db, min_support=3, max_groups=10)
    assert groups == []
    below = candidate_groups(db, min_support=3, max_groups=10, include_below_support=True)
    assert len(below) == 1 and below[0].eligible is False


def test_new_evidence_gate_skips_consolidated_groups(db):
    for i in (1, 2, 3):
        _episode(db, i, ["java"], [])
    db.upsert_consolidation_state(
        value="java",
        representative_kind="entity",
        last_consolidated_at="2026-09-02T00:00:00+00:00",
        episode_count=3, semantic_id=None,
    )
    assert candidate_groups(db, min_support=3, max_groups=10) == []

    _episode(db, 4, ["java"], [])
    groups = candidate_groups(db, min_support=3, max_groups=10)
    assert len(groups) == 1
    assert groups[0].episode_count == 4
    assert groups[0].new_evidence == 1


def test_state_is_value_scoped_across_kinds(db):
    """topic:java and entity:java are one concept — consolidating one must
    suppress the other (regression: state was per (kind, value) and the
    un-tracked kind re-entered with full 'new evidence')."""
    for i in (1, 2, 3):
        _episode(db, i, ["java"], ["java"])
    db.upsert_consolidation_state(
        value="java",
        representative_kind="entity",
        last_consolidated_at="2026-09-02T00:00:00+00:00",
        episode_count=3, semantic_id=None,
    )
    assert candidate_groups(db, min_support=3, max_groups=10) == []


def test_entity_and_topic_collapse_to_one_group(db):
    for i in (1, 2, 3):
        _episode(db, i, ["minecraft"], ["minecraft"])
    groups = candidate_groups(db, min_support=3, max_groups=10)
    assert len(groups) == 1
    assert groups[0].kind == "entity"  # entity wins the collapse


def test_ordering_and_max_groups(db):
    for i in (1, 2, 3, 4):
        _episode(db, i, ["java"], [])
    for i in (5, 6, 7):
        _episode(db, i, ["rust"], [])
    groups = candidate_groups(db, min_support=3, max_groups=1)
    assert len(groups) == 1
    assert groups[0].value == "java"  # larger group first
