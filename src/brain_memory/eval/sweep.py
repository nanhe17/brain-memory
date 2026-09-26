"""权重扫参：在 7 因子单纯形中搜索更好的检索权重。

对权重单纯形做固定种子的 Dirichlet 随机采样（外加当前默认）——7 个
维度上网格搜索没有希望，随机搜索可复现。排序键：平均 Recall@k、
然后 MRR、然后更少的违规数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_memory.config import MemoryConfig, RetrievalWeights
from brain_memory.eval.report import EvalReport
from brain_memory.eval.scenario import ScenarioFile, ScenarioRunner

_FACTOR_FIELDS = ("semantic", "keyword", "recency", "importance", "frequency", "entity", "context")


@dataclass
class SweepCandidate:
    """一组权重及其评测报告。"""

    weights: RetrievalWeights
    report: EvalReport

    @property
    def sort_key(self) -> tuple[float, float, int]:
        """排序键（recall, mrr, -violations）。"""
        return (self.report.recall, self.report.mrr, -self.report.violations)


def _weights_from_vector(vector: np.ndarray) -> RetrievalWeights:
    """单纯形向量 -> 权重对象。"""
    return RetrievalWeights(**dict(zip(_FACTOR_FIELDS, (float(x) for x in vector))))


def sample_weights(rng: np.random.Generator) -> RetrievalWeights:
    """从权重单纯形随机采样一组权重。"""
    return _weights_from_vector(rng.dirichlet(np.ones(len(_FACTOR_FIELDS))))


def sweep(
    scenarios: list[ScenarioFile],
    base_config: MemoryConfig,
    *,
    samples: int = 40,
    k: int = 5,
    seed: int = 2026,
) -> list[SweepCandidate]:
    """评估当前权重外加 ``samples`` 组随机权重向量。"""
    rng = np.random.default_rng(seed)
    candidates: list[SweepCandidate] = []

    weight_sets: list[RetrievalWeights] = [base_config.weights]
    weight_sets += [sample_weights(rng) for _ in range(samples)]

    for weights in weight_sets:
        run_config = base_config.model_copy(update={"weights": weights})
        own_runner = ScenarioRunner(config=run_config, default_k=k)
        results = [own_runner.run(scenario) for scenario in scenarios]
        report = EvalReport.build(weights, k, results)
        candidates.append(SweepCandidate(weights=weights, report=report))

    candidates.sort(key=lambda c: c.sort_key, reverse=True)
    return candidates


def export_env(weights: RetrievalWeights) -> str:
    """把权重导出为 ``MEMORY_W_*`` export 行。"""
    names = {
        "semantic": "MEMORY_W_SEMANTIC",
        "keyword": "MEMORY_W_KEYWORD",
        "recency": "MEMORY_W_RECENCY",
        "importance": "MEMORY_W_IMPORTANCE",
        "frequency": "MEMORY_W_FREQUENCY",
        "entity": "MEMORY_W_ENTITY",
        "context": "MEMORY_W_CONTEXT",
    }
    return "\n".join(f"export {names[field]}={getattr(weights, field):.4f}" for field in _FACTOR_FIELDS)
