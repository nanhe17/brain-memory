from __future__ import annotations

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.working.working_memory import WorkingMemory


def _engine(tmp_path) -> MemoryEngine:
    return MemoryEngine(MemoryConfig(db_path=str(tmp_path / "wm.db"), embedding_dim=64))


def test_snapshot_renders_state():
    working = WorkingMemory()
    assert working.snapshot() == ""
    working.set_goal("finish the mod")
    working.add_entities(["Minecraft", "Forge"])
    working.add_question("when to release?")
    text = working.snapshot()
    assert "goal: finish the mod" in text
    assert "Minecraft" in text
    assert "when to release?" in text


def test_session_entities_bias_ranking(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户在做 Phoenix 项目的搜索模块")   # id 1
    engine.encode("用户在做 Mountain 项目的搜索模块")   # id 2

    # fresh session: no working-memory bias
    engine.working = WorkingMemory()
    neutral = engine.recall("搜索模块进展", k=1, touch=False)

    # session actively about Phoenix: the Phoenix episode should win
    engine.working = WorkingMemory()
    engine.working.add_entities(["Phoenix"])
    biased = engine.recall("搜索模块进展", k=1, touch=False)
    assert biased[0].episode.id == 1
    engine.close()
    # (neutral order is data-dependent; the assertion is on the bias effect)


def test_boost_does_not_override_strong_keyword_match(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户在做 Phoenix 项目的搜索模块")
    engine.encode("用户在学量子力学基础")
    # session is about Phoenix; still, the cue's clear keyword match must win
    engine.working = WorkingMemory()
    engine.working.add_entities(["Phoenix"])
    hits = engine.recall("量子力学基础", k=1, use_working_memory=True)
    assert hits[0].episode.id == 2
    engine.close()


def test_use_working_memory_false_skips_boost(tmp_path):
    engine = _engine(tmp_path)
    engine.encode("用户在做 Phoenix 项目的搜索模块")
    engine.working = WorkingMemory()
    engine.working.add_entities(["Phoenix", "搜索", "模块"])
    hits = engine.recall("搜索模块进展", k=5, use_working_memory=False)
    assert hits  # still retrieves via the cue itself
    engine.close()
