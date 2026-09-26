from __future__ import annotations

from brain_memory.benchmark.long_horizon import format_markdown, generate_episodes, run


def test_generator_is_deterministic_and_plants_facts():
    first, planted_first = generate_episodes(100, seed=7)
    second, planted_second = generate_episodes(100, seed=7)
    assert first == second
    assert len(planted_first) == 10  # every 10th episode plants a codename
    assert all("请记住" in first[index] for index, _ in planted_first)


def test_run_smoke_at_small_scale():
    report = run(n_episodes=50, n_queries=10, seed=2026)
    # all measurement sections present
    for key in ("encode_seconds", "recall_p50_ms", "recall_p95_ms",
                "planted_recall_at_5", "consolidate_seconds", "decay_seconds",
                "db_size_mb", "neighborhood_avg_ms", "ppr_seconds"):
        assert key in report
    # planted facts are recallable through the whole pipeline
    assert report["planted_recall_at_5"] > 0.5
    assert report["semantic_memories_created"] > 0


def test_markdown_render_contains_metrics():
    report = run(n_episodes=30, n_queries=6, seed=1)
    text = format_markdown(report)
    assert text.startswith("# Long-horizon benchmark")
    assert "planted-fact Recall@5" in text
