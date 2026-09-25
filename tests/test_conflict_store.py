from __future__ import annotations

import pytest

from brain_memory.consolidation.conflicts import ConflictStore
from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.embeddings.hash_embedder import HashEmbedder
from brain_memory.models import ConflictKind, ConflictStatus, SemanticKind
from brain_memory.storage.db import Database


@pytest.fixture()
def stores(tmp_path):
    db = Database(str(tmp_path / "conflicts.db"))
    semantic = SemanticStore(db)
    yield ConflictStore(db), semantic, db
    db.close()


def _semantic(semantic: SemanticStore):
    return semantic.create(
        concept="java",
        kind=SemanticKind.CO_OCCURRENCE,
        statement="java 在 3 条记忆中反复出现",
        confidence=0.5,
        evidence_ids=[1, 2, 3],
        embedding=HashEmbedder(dim=16).embed_texts(["java"])[0],
    )


def test_create_and_resolve(stores):
    conflicts, semantic, db = stores
    memory = _semantic(semantic)
    conflict = conflicts.create(
        semantic_id=memory.id,
        kind=ConflictKind.UNRESOLVED,
        old_version=memory.version,
        statement_before=memory.statement,
        trigger_episode_id=9,
        trigger_kind="encode",
    )
    assert conflict.status is ConflictStatus.OPEN
    assert conflicts.open_semantic_ids() == [memory.id]

    touched = conflicts.resolve_for_semantic(
        memory.id, kind=ConflictKind.EVOLUTION, resolution_version=2
    )
    assert touched == 1
    resolved = conflicts.for_semantic(memory.id)[0]
    assert resolved.status is ConflictStatus.RESOLVED
    assert resolved.kind is ConflictKind.EVOLUTION
    assert resolved.resolution_version == 2
    assert conflicts.open_semantic_ids() == []


def test_dismiss_keeps_kind(stores):
    conflicts, semantic, db = stores
    memory = _semantic(semantic)
    conflicts.create(
        semantic_id=memory.id,
        kind=ConflictKind.UNRESOLVED,
        old_version=1,
        statement_before="old",
        trigger_episode_id=None,
        trigger_kind="consolidation",
    )
    conflicts.resolve_for_semantic(
        memory.id, kind=ConflictKind.UNRESOLVED, resolution_version=2, dismiss=True
    )
    dismissed = conflicts.for_semantic(memory.id)[0]
    assert dismissed.status is ConflictStatus.DISMISSED
    assert dismissed.kind is ConflictKind.UNRESOLVED  # kind untouched on dismiss


def test_list_by_status(stores):
    conflicts, semantic, db = stores
    memory = _semantic(semantic)
    conflicts.create(
        semantic_id=memory.id, kind=ConflictKind.UNRESOLVED, old_version=1,
        statement_before=None, trigger_episode_id=None, trigger_kind="encode",
    )
    conflicts.resolve_for_semantic(memory.id, kind=ConflictKind.EVOLUTION, resolution_version=2)
    assert len(conflicts.list()) == 1
    assert len(conflicts.list(status="resolved")) == 1
    assert conflicts.list(status="open") == []
