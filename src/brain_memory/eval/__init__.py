"""Retrieval evaluation harness: scenarios, metrics, reports, weight sweeps.

The harness answers "did this change actually improve retrieval?" (design doc
§33-35).  Scenarios are YAML files; each encodes episodes into a fresh
in-memory engine, fires cues, and scores the ranked results.  It is
embedder-agnostic: CI runs it on the hash embedder, real tuning runs with a
cloud provider configured via the usual MEMORY_* env vars.
"""

from brain_memory.eval.metrics import (
    mrr,
    precision_at_k,
    recall_at_k,
    violation_count,
)
from brain_memory.eval.report import EvalReport, ScenarioReport, TagReport
from brain_memory.eval.scenario import (
    CueResult,
    ScenarioCue,
    ScenarioEpisode,
    ScenarioFile,
    ScenarioResult,
    ScenarioRunner,
    load_scenarios,
)
from brain_memory.eval.sweep import SweepCandidate, export_env, sweep

__all__ = [
    "ScenarioFile",
    "ScenarioCue",
    "ScenarioEpisode",
    "ScenarioResult",
    "CueResult",
    "ScenarioRunner",
    "load_scenarios",
    "EvalReport",
    "ScenarioReport",
    "TagReport",
    "SweepCandidate",
    "sweep",
    "export_env",
    "recall_at_k",
    "mrr",
    "precision_at_k",
    "violation_count",
]
