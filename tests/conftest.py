from __future__ import annotations

import pytest

from brain_memory import MemoryConfig, MemoryEngine


@pytest.fixture()
def config(tmp_path):
    return MemoryConfig(
        db_path=str(tmp_path / "memory.db"),
        embedding_dim=64,
        candidate_pool_per_channel=16,
    )


@pytest.fixture()
def engine(config):
    engine = MemoryEngine(config)
    yield engine
    engine.close()
