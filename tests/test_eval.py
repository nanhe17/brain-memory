from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from brain_memory.config import MemoryConfig
from brain_memory.eval import (
    EvalReport,
    ScenarioFile,
    ScenarioRunner,
    load_scenarios,
    mrr,
    precision_at_k,
    recall_at_k,
    violation_count,
)

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "scenarios"

HASH_CONFIG = MemoryConfig(db_path=":memory:", embedding_dim=256)


# -- metrics ------------------------------------------------------------------


def test_recall_at_k():
    assert recall_at_k({"e1", "e2"}, ["e3", "e1", "e2"], k=3) == 1.0
    assert recall_at_k({"e1", "e2"}, ["e3", "e1", "e4"], k=2) == 0.5
    assert recall_at_k(set(), ["e1"], k=1) == 1.0  # nothing expected -> perfect


def test_mrr():
    assert mrr({"e2"}, ["e1", "e2", "e3"]) == pytest.approx(0.5)
    assert mrr({"e3"}, ["e3", "e2", "e1"]) == pytest.approx(1.0)
    assert mrr({"e9"}, ["e1", "e2", "e3"]) == 0.0
    assert mrr({"s1"}, ["e1", "s1"]) == pytest.approx(0.5)  # semantic tags mix in


def test_precision_at_k():
    assert precision_at_k({"e1", "e2"}, ["e1", "e2", "e3", "e4"], k=4) == pytest.approx(0.5)
    assert precision_at_k({"e1"}, [], k=2) == 0.0


def test_violation_count():
    assert violation_count({"e2"}, ["e2", "e1"], k=2) == 1
    assert violation_count({"e2"}, ["e1", "e2"], k=1) == 0


# -- scenario loading ----------------------------------------------------------


def _write_scenario(tmp_path: Path, data: dict, filename: str = "s.yaml") -> Path:
    import yaml

    path = tmp_path / filename
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


MINIMAL = {
    "name": "tiny",
    "tags": ["recall"],
    "episodes": [{"text": "用户在写 Rust 异步服务"}, {"text": "用户在研究量子化学"}],
    "cues": [{"text": "那个 Rust 服务", "expect": [0], "k": 1}],
}


def test_load_scenarios_sorted_and_validated(tmp_path):
    _write_scenario(tmp_path, MINIMAL, "b.yaml")
    _write_scenario(tmp_path, MINIMAL | {"name": "alpha"}, "a.yaml")
    loaded = load_scenarios(tmp_path)
    assert [s.name for s in loaded] == ["alpha", "tiny"]


def test_load_rejects_out_of_range_indexes(tmp_path):
    bad = {
        "name": "bad",
        "episodes": [{"text": "one"}],
        "cues": [{"text": "cue", "expect": [0, 5]}],
    }
    _write_scenario(tmp_path, bad)
    with pytest.raises(ValidationError):
        load_scenarios(tmp_path)


def test_load_rejects_empty_expect(tmp_path):
    bad = {
        "name": "bad",
        "episodes": [{"text": "one"}],
        "cues": [{"text": "cue", "expect": [], "expect_concepts": []}],
    }
    _write_scenario(tmp_path, bad)
    with pytest.raises(ValidationError):
        load_scenarios(tmp_path)


def test_load_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scenarios(tmp_path / "nope")


# -- runner ---------------------------------------------------------------------


def test_runner_maps_indexes_and_scores(tmp_path):
    scenario = ScenarioFile.model_validate(MINIMAL)
    result = ScenarioRunner(config=HASH_CONFIG, default_k=2).run(scenario)
    assert result.name == "tiny"
    cue = result.cues[0]
    assert cue.expected_ids == ["e1"]  # index 0 -> id 1
    assert cue.recall == 1.0
    assert cue.mrr == 1.0
    assert cue.violations == 0


def test_runner_days_ago_sets_recency():
    scenario = ScenarioFile.model_validate(
        {
            "name": "recency",
            "episodes": [
                {"text": "用户喜欢 A 项目", "days_ago": 60},
                {"text": "用户最近迷上了 A 项目的新特性", "days_ago": 0},
            ],
            "cues": [{"text": "用户对 A 项目最近的热情", "expect": [1], "k": 1}],
        }
    )
    result = ScenarioRunner(config=HASH_CONFIG).run(scenario)
    assert result.cues[0].ranked_ids[0] == "e2"


def test_runner_consolidation_and_concepts():
    scenario = ScenarioFile.model_validate(
        {
            "name": "consolidated",
            "tags": ["consolidation"],
            "consolidate": True,
            "episodes": [
                {"text": "用户在学习 Java 后端", "days_ago": 10},
                {"text": "用户喜欢自学原理，正在学 Java", "days_ago": 8},
                {"text": "用户整理了 Java 的知识图谱", "days_ago": 6},
                {"text": "用户今天去爬山了", "days_ago": 1},
            ],
            "cues": [{"text": "用户的 Java 学习情况", "expect_concepts": ["java"], "k": 5}],
        }
    )
    result = ScenarioRunner(config=HASH_CONFIG).run(scenario)
    cue = result.cues[0]
    assert any(tag.startswith("s") for tag in cue.expected_ids), "semantic tag expected"
    assert cue.recall == 1.0


# -- packaged seed scenarios (baseline regression canaries) ---------------------


@pytest.fixture(scope="module")
def seed_report():
    scenarios = load_scenarios(SCENARIOS_DIR)
    runner = ScenarioRunner(config=HASH_CONFIG, default_k=5)
    results = [runner.run(s) for s in scenarios]
    return EvalReport.build(HASH_CONFIG.weights, 5, results)


def test_seed_scenario_count(seed_report):
    assert len(seed_report.scenarios) == 10


def test_seed_no_forbidden_leaks(seed_report):
    assert seed_report.violations == 0


def test_seed_full_recall(seed_report):
    assert seed_report.recall == 1.0


def test_seed_tag_breakdown(seed_report):
    by_tag = {t.tag: t for t in seed_report.by_tag()}
    assert {"recall", "separation", "temporal", "contamination"} <= set(by_tag)
