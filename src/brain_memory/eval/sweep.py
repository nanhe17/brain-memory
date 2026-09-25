"""Weight sweeps: search the 7-factor simplex for better retrieval weights.

Random search over a Dirichlet sample of the weight simplex (plus the current
default) — with 7 factors, grid search is dead on arrival, and random search
with a fixed seed is reproducible.  Ranking key: mean recall@k, then MRR,
then fewer violations.
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
    weights: RetrievalWeights
    report: EvalReport

    @property
    def sort_key(self) -> tuple[float, float, int]:
        return (self.report.recall, self.report.mrr, -self.report.violations)


def _weights_from_vector(vector: np.ndarray) -> RetrievalWeights:
    return RetrievalWeights(**dict(zip(_FACTOR_FIELDS, (float(x) for x in vector))))


def sample_weights(rng: np.random.Generator) -> RetrievalWeights:
    return _weights_from_vector(rng.dirichlet(np.ones(len(_FACTOR_FIELDS))))


def sweep(
    scenarios: list[ScenarioFile],
    base_config: MemoryConfig,
    *,
    samples: int = 40,
    k: int = 5,
    seed: int = 2026,
) -> list[SweepCandidate]:
    """Evaluate the current weights plus ``samples`` random weight vectors."""
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
    """``MEMORY_W_*`` export lines for a weight vector."""
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
