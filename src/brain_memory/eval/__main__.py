"""检索评测 harness 的 CLI。

示例::

    brain-memory-eval --scenarios benchmarks/scenarios
    brain-memory-eval --scenarios benchmarks/scenarios --sweep 60
    brain-memory-eval --scenarios benchmarks/scenarios --report report.md

harness 按你的环境原样运行：配置了云嵌入 provider 就度量真实检索质量；
没有则运行确定性 hash 嵌入器（适合 CI 回归，不适合调参）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brain_memory.config import MemoryConfig
from brain_memory.eval.report import EvalReport
from brain_memory.eval.scenario import ScenarioRunner, load_scenarios
from brain_memory.eval.sweep import export_env, sweep


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：评测 / 扫参 / 写报告。"""
    parser = argparse.ArgumentParser(
        prog="brain-memory-eval",
        description="Evaluate memory retrieval quality over YAML scenarios.",
    )
    parser.add_argument(
        "--scenarios",
        default="./benchmarks/scenarios",
        help="directory containing scenario YAML files (default: %(default)s)",
    )
    parser.add_argument("--k", type=int, default=5, help="default top-k per cue (default: %(default)s)")
    parser.add_argument(
        "--sweep", type=int, default=0, metavar="N",
        help="also try N random weight vectors and report the best (default: off)",
    )
    parser.add_argument("--seed", type=int, default=2026, help="sweep random seed")
    parser.add_argument("--report", type=Path, default=None, help="write a markdown report to this path")
    args = parser.parse_args(argv)

    try:
        scenarios = load_scenarios(args.scenarios)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not scenarios:
        print(f"error: no scenario files in {args.scenarios}", file=sys.stderr)
        return 2

    base_config = MemoryConfig.from_env()
    runner = ScenarioRunner(config=base_config, default_k=args.k)
    print(f"embedder={base_config.embedding_provider} scenarios={len(scenarios)} k={args.k}")

    results = [runner.run(scenario) for scenario in scenarios]
    report = EvalReport.build(base_config.weights, args.k, results)
    print()
    print(report.to_markdown(title="Baseline (current config)"))

    # cue 级失败明细（含语义版本断言失败）
    misses = [
        (result.name, cue)
        for result in results
        for cue in result.cues
        if cue.recall < 1.0 or cue.violations > 0 or cue.semantic_version_ok is False
    ]
    if misses:
        print("## Cue misses")
        print()
        for scenario_name, cue in misses:
            expected = ",".join(str(i) for i in cue.expected_ids)
            got = ",".join(str(i) for i in cue.ranked_ids)
            print(
                f"- [{scenario_name}] '{cue.cue}' expected [{expected}] got [{got}] "
                f"(recall={cue.recall:.2f}, violations={cue.violations}, "
                f"version_ok={cue.semantic_version_ok})"
            )
        print()

    if args.sweep > 0:
        print(f"Sweeping {args.sweep} random weight vectors (seed={args.seed})…")
        ranked = sweep(scenarios, base_config, samples=args.sweep, k=args.k, seed=args.seed)
        print()
        print("## Sweep top 5")
        print()
        print("| # | recall | mrr | violations | weights |")
        print("|---|---|---|---|---|")
        for i, candidate in enumerate(ranked[:5], 1):
            w = candidate.weights
            weight_str = " ".join(f"{field}={getattr(w, field):.2f}" for field in
                                  ("semantic", "keyword", "recency", "importance",
                                   "frequency", "entity", "context"))
            print(
                f"| {i} | {candidate.report.recall:.4f} | {candidate.report.mrr:.4f} "
                f"| {candidate.report.violations} | {weight_str} |"
            )
        best = ranked[0]
        print()
        print("Best configuration as env exports:")
        print()
        print("```bash")
        print(export_env(best.weights))
        print("```")
        print()
        if args.report is not None:
            (args.report.parent if args.report.parent != Path("") else Path(".")).mkdir(
                parents=True, exist_ok=True
            )
            args.report.write_text(
                best.report.to_markdown(title=f"Best sweep config (seed={args.seed})"),
                encoding="utf-8",
            )
            print(f"best-config report written to {args.report}")
    elif args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_markdown(), encoding="utf-8")
        print(f"report written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
