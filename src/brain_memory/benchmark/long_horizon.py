"""Long-horizon scale benchmark (design doc, Benchmark 5).

Generates deterministic synthetic memories at a chosen scale, then measures
what actually matters operationally: encode throughput, recall latency
(p50/p95), planted-fact Recall@k, store size, decay/consolidation/neighborhood
costs.  With the hash embedder this is a latency-and-scale instrument; point
it at a cloud embedding provider for a quality baseline too.

CLI::

    brain-memory-benchmark --episodes 1000 --queries 50 --report out.md

Deliberately not a pytest: perf numbers do not belong in CI assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine

_TOPICS = [
    ("Java 后端", "java backend"),
    ("Minecraft 模组", "minecraft modding"),
    ("Rust 异步", "rust async"),
    ("数据管道", "data pipelines"),
    ("论文阅读", "paper reading"),
    ("前端重构", "frontend refactoring"),
    ("K8s 运维", "k8s ops"),
    ("机器学习训练", "ml training"),
]

_DETAILS = [
    "推进了核心模块的设计与实现",
    "排查了一个难以复现的问题",
    "和团队讨论了下一步的方案",
    "重构了一段历史遗留代码",
    "阅读文档并整理了笔记",
    "fixed a flaky test and documented the root cause",
    "profiled the hot path and removed an allocation",
    "drafted a design note for the next iteration",
]


def generate_episodes(n: int, seed: int = 2026) -> tuple[list[str], list[tuple[int, str]]]:
    """Deterministic synthetic stream: topic clusters + junk + a planted
    high-importance fact every 10 episodes (``proj-<index>`` codenames)."""
    rng = random.Random(seed)
    episodes: list[str] = []
    planted: list[tuple[int, str]] = []
    for i in range(n):
        if i % 10 == 0:
            codename = f"proj-{i:05d}"
            episodes.append(f"请记住：用户的项目代号是 {codename}，这个项目很重要")
            planted.append((i, codename))
            continue
        topic_zh, topic_en = _TOPICS[i % len(_TOPICS)]
        detail = rng.choice(_DETAILS)
        if i % 3 == 0:
            episodes.append(f"{topic_en} session {i}: {detail}")
        else:
            episodes.append(f"{topic_zh}：第 {i} 条工作记录，{detail}")
    return episodes, planted


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[idx]


def run(n_episodes: int = 1000, n_queries: int = 50, seed: int = 2026,
        db_path: str | None = None) -> dict:
    """Run the benchmark and return a JSON-serializable report."""
    tmp_dir = None
    if db_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="brain-memory-bench-")
        db_path = os.path.join(tmp_dir, "bench.db")
    config = MemoryConfig(db_path=db_path, embedding_dim=256)
    engine = MemoryEngine(config)
    report: dict = {"episodes": n_episodes, "queries": n_queries, "seed": seed}

    try:
        episodes, planted = generate_episodes(n_episodes, seed)
        started = time.perf_counter()
        for text in episodes:
            engine.encode(text)
        report["encode_seconds"] = round(time.perf_counter() - started, 3)
        report["encode_per_1k_seconds"] = round(
            report["encode_seconds"] / max(n_episodes, 1) * 1000, 3
        )

        first_started = time.perf_counter()
        engine.recall(episodes[0][:12], k=1)
        report["first_recall_seconds"] = round(time.perf_counter() - first_started, 4)

        rng = random.Random(seed + 1)
        sampled = rng.sample(planted, min(n_queries // 2, len(planted)))
        recall_hits = 0
        latencies: list[float] = []
        for index, codename in sampled:
            cue = f"{codename} 的代号是什么"
            started = time.perf_counter()
            hits = engine.recall(cue, k=5, use_working_memory=False)
            latencies.append(time.perf_counter() - started)
            if any(codename in h.episode.content for h in hits):
                recall_hits += 1
        topic_queries = max(1, n_queries - len(sampled))
        for _ in range(topic_queries):
            topic_zh, _topic_en = rng.choice(_TOPICS)
            started = time.perf_counter()
            engine.recall(topic_zh, k=5, use_working_memory=False)
            latencies.append(time.perf_counter() - started)
        report["recall_p50_ms"] = round(statistics.median(latencies) * 1000, 3)
        report["recall_p95_ms"] = round(_pctl(latencies, 0.95) * 1000, 3)
        report["planted_recall_at_5"] = round(
            recall_hits / max(len(sampled), 1), 4
        )

        started = time.perf_counter()
        consolidation = engine.consolidate(max_groups=20, max_episodes_per_group=30)
        report["consolidate_seconds"] = round(time.perf_counter() - started, 3)
        report["semantic_memories_created"] = len(consolidation.created)

        started = time.perf_counter()
        engine.decay()
        report["decay_seconds"] = round(time.perf_counter() - started, 4)

        started = time.perf_counter()
        for probe in range(20):
            engine.neighborhood(f"e{1 + probe % max(n_episodes, 1)}")
        report["neighborhood_avg_ms"] = round(
            (time.perf_counter() - started) / 20 * 1000, 3
        )

        started = time.perf_counter()
        engine.graph_rank("Java 后端 的进展", k=10)
        report["ppr_seconds"] = round(time.perf_counter() - started, 4)

        stats = engine.stats()
        report["db_size_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 3)
        report["active_episodes"] = stats.active
        report["semantic_memories"] = stats.semantic_memories
        report["distinct_entities"] = stats.distinct_entities
    finally:
        engine.close()
        if tmp_dir:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
    return report


def format_markdown(report: dict) -> str:
    lines = [
        "# Long-horizon benchmark",
        "",
        f"episodes: **{report['episodes']}** · queries: **{report['queries']}** · "
        f"seed: **{report['seed']}**",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    labels = {
        "encode_seconds": "encode total (s)",
        "encode_per_1k_seconds": "encode per 1k episodes (s)",
        "first_recall_seconds": "first recall / index build (s)",
        "recall_p50_ms": "recall p50 (ms)",
        "recall_p95_ms": "recall p95 (ms)",
        "planted_recall_at_5": "planted-fact Recall@5",
        "consolidate_seconds": "consolidate (s)",
        "semantic_memories_created": "semantic memories created",
        "decay_seconds": "decay sweep (s)",
        "neighborhood_avg_ms": "neighborhood avg (ms)",
        "ppr_seconds": "graph_rank / PPR (s)",
        "db_size_mb": "db size (MB)",
        "active_episodes": "active episodes (post-decay)",
        "semantic_memories": "semantic memories",
        "distinct_entities": "distinct entities",
    }
    for key, label in labels.items():
        if key in report:
            lines.append(f"| {label} | {report[key]} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brain-memory-benchmark",
        description="Long-horizon scale benchmark (doc Benchmark 5).",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--db", default=None, help="sqlite path (default: temp file)")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the markdown report to this path")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead")
    args = parser.parse_args(argv)

    report = run(args.episodes, args.queries, args.seed, db_path=args.db)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_markdown(report))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(format_markdown(report), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
