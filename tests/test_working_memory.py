from __future__ import annotations

from brain_memory.working.working_memory import WorkingMemory, estimate_tokens


def test_estimate_tokens_mixed_script():
    assert estimate_tokens("hello") >= 1
    assert estimate_tokens("你好") == 2
    assert estimate_tokens("") == 0


def test_prompt_block_contains_goal_and_memories(engine):
    engine.encode("用户正在开发 Minecraft P-51 飞机模组")
    engine.recall("飞机模组", k=1)
    engine.working.set_goal("continue the airplane mod")
    block = engine.working.build_prompt_block()
    assert "[GOAL] continue the airplane mod" in block
    assert "[RELEVANT MEMORIES]" in block
    assert "#1" in block
    assert "<why:" in block


def test_prompt_block_respects_budget_by_dropping_not_truncating():
    working = WorkingMemory(token_budget=300)
    engine_like_hits = []
    for i in range(20):
        from brain_memory.models import Episode, FactorScores, RecallResult
        from datetime import datetime, timezone

        episode = Episode(
            id=i,
            content=f"memory {i} " + "x" * 200,
            content_hash=f"h{i}",
            created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            importance=0.5,
            confidence=0.5,
        )
        engine_like_hits.append(RecallResult(episode=episode, score=0.5,
                                             factors=FactorScores(), reasons=["recency"]))
    working.remember_recall(engine_like_hits)
    block = working.build_prompt_block()
    assert estimate_tokens(block) <= 300
    emitted = block.count("- #")
    assert 0 < emitted < 20, "some memories fit, the rest are dropped"
    for line in block.splitlines():
        if line.startswith("- #"):
            assert line.endswith(">"), "entries are never cut mid-memory"


def test_state_tracking(engine):
    engine.encode("用户在用 Redis 做缓存")
    engine.recall("Redis 缓存", k=1)
    state = engine.working.state
    assert state.retrieved_memory_ids == [1]
    assert "Redis" in state.active_entities or "redis" in [e.casefold() for e in state.active_entities]


def test_resolve_question():
    working = WorkingMemory()
    working.add_question("什么时候发布？")
    working.add_question("什么时候发布？")  # deduped
    assert len(working.state.unresolved_questions) == 1
    working.resolve_question("什么时候发布？")
    assert working.state.unresolved_questions == []
