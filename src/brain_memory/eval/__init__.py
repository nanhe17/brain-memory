"""检索评测 harness：场景、指标与权重扫参。

harness 回答"这次改动真的提升了检索吗？"（设计文档 §33-35）。场景是
YAML 文件；每个场景把 episode 编码进一个全新的内存引擎，触发 cue 并
对排序结果打分。它对嵌入器不可知：CI 用 hash 嵌入器跑，真实调参配置
云 provider（经常规 MEMORY_* 环境变量）后运行。
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
