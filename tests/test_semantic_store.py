from __future__ import annotations

import numpy as np
import pytest

from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.models import SemanticKind
from brain_memory.storage.db import Database


@pytest.fixture()
def store(tmp_path):
    db = Database(str(tmp_path / "sem.db"))
    yield SemanticStore(db), db
    db.close()


def _vec(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_create_writes_v1_and_version_row(store):
    semantic, db = store
    memory = semantic.create(
        concept="java",
        kind=SemanticKind.CO_OCCURRENCE,
        statement="java 在 3 条记忆中反复出现",
        confidence=0.5,
        evidence_ids=[1, 2, 3],
        embedding=_vec(1),
    )
    assert memory.version == 1
    assert memory.concept == "java"
    assert semantic.versions(memory.id)[0].change_reason == "initial consolidation"
    assert semantic.get_by_concept("JAVA") is not None  # concept key casefolded


def test_update_bumps_version_and_merges_history(store):
    semantic, db = store
    memory = semantic.create(
        concept="java",
        kind=SemanticKind.CO_OCCURRENCE,
        statement="v1 statement",
        confidence=0.4,
        evidence_ids=[1],
        embedding=_vec(1),
    )
    updated = semantic.update(
        memory,
        kind=SemanticKind.PREFERENCE,
        statement="v2 statement",
        confidence=0.6,
        evidence_ids=[1, 2],
        embedding=_vec(2),
        change_reason="new evidence",
    )
    assert updated.version == 2
    assert updated.statement == "v2 statement"
    versions = semantic.versions(memory.id)
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].statement == "v1 statement"
    assert versions[1].change_reason == "new evidence"


def test_fts_search_finds_statement(store):
    semantic, db = store
    memory = semantic.create(
        concept="java",
        kind=SemanticKind.CO_OCCURRENCE,
        statement="java 在 3 条记忆中反复出现，主要涉及 spring、学习",
        confidence=0.5,
        evidence_ids=[1, 2, 3],
        embedding=_vec(1),
    )
    hits = semantic.fts_search('"java"', 5)
    assert [m.id for m, _rank in hits] == [memory.id]
    assert semantic.fts_search('"量子化学"', 5) == []


def test_touch_and_status(store):
    semantic, db = store
    memory = semantic.create(
        concept="rust",
        kind=SemanticKind.CO_OCCURRENCE,
        statement="rust shows up across 3 memories",
        confidence=0.44,
        evidence_ids=[7, 8, 9],
        embedding=_vec(3),
    )
    semantic.touch([memory.id])
    assert semantic.get(memory.id).access_count == 1
    assert semantic.archive(memory.id) is True
    assert semantic.get(memory.id).status.value == "archived"
    assert semantic.fts_search('"rust"', 5) == []  # archived drops out of FTS
    assert semantic.restore(memory.id) is True
    assert semantic.fts_search('"rust"', 5)


def test_statement_update_resyncs_fts(store):
    semantic, db = store
    memory = semantic.create(
        concept="docker",
        kind=SemanticKind.CO_OCCURRENCE,
        statement="old about 容器部署",
        confidence=0.5,
        evidence_ids=[1],
        embedding=_vec(4),
    )
    semantic.update(
        memory,
        kind=SemanticKind.FACT,
        statement="new about kubernetes 迁移",
        confidence=0.6,
        evidence_ids=[1, 2],
        embedding=_vec(5),
        change_reason="revised",
    )
    assert semantic.fts_search('"容器部署"', 5) == []
    assert semantic.fts_search('"kubernetes"', 5)
