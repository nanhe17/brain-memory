"""聚合评测报告：标签分组与 markdown 输出。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from brain_memory.config import RetrievalWeights
from brain_memory.eval.scenario import ScenarioResult


class ScenarioReport(BaseModel):
    """单场景的报告行。"""

    name: str
    tags: list[str]
    cues: int
    recall: float
    mrr: float
    violations: int


class TagReport(BaseModel):
    """同标签场景的聚合报告。"""

    tag: str
    scenarios: int
    recall: float
    mrr: float
    violations: int


class EvalReport(BaseModel):
    """一个配置在全部场景上的聚合结果。"""

    weights: RetrievalWeights
    k: int
    scenarios: list[ScenarioReport] = Field(default_factory=list)

    @classmethod
    def build(cls, weights: RetrievalWeights, k: int, results: list[ScenarioResult]) -> "EvalReport":
        """从场景结果构建报告。"""
        return cls(
            weights=weights,
            k=k,
            scenarios=[
                ScenarioReport(
                    name=r.name,
                    tags=r.tags,
                    cues=len(r.cues),
                    recall=round(r.recall, 4),
                    mrr=round(r.mrr, 4),
                    violations=r.violations,
                )
                for r in results
            ],
        )

    @property
    def recall(self) -> float:
        """全部场景 recall 均值。"""
        if not self.scenarios:
            return 0.0
        return sum(s.recall for s in self.scenarios) / len(self.scenarios)

    @property
    def mrr(self) -> float:
        """全部场景 MRR 均值。"""
        if not self.scenarios:
            return 0.0
        return sum(s.mrr for s in self.scenarios) / len(self.scenarios)

    @property
    def violations(self) -> int:
        """全部禁用泄漏之和。"""
        return sum(s.violations for s in self.scenarios)

    def by_tag(self) -> list[TagReport]:
        """按标签聚合（无标签的场景归入 untagged）。"""
        buckets: dict[str, list[ScenarioReport]] = {}
        for scenario in self.scenarios:
            for tag in scenario.tags or ["untagged"]:
                buckets.setdefault(tag, []).append(scenario)
        reports = [
            TagReport(
                tag=tag,
                scenarios=len(items),
                recall=round(sum(i.recall for i in items) / len(items), 4),
                mrr=round(sum(i.mrr for i in items) / len(items), 4),
                violations=sum(i.violations for i in items),
            )
            for tag, items in buckets.items()
        ]
        reports.sort(key=lambda r: r.tag)
        return reports

    def to_markdown(self, title: str = "Evaluation report") -> str:
        """渲染为 markdown 报告。"""
        lines = [
            f"# {title}",
            "",
            f"Recall@{self.k}: **{self.recall:.4f}** · MRR: **{self.mrr:.4f}** · "
            f"violations: **{self.violations}**",
            "",
            "## Scenarios",
            "",
            "| scenario | tags | cues | recall | mrr | violations |",
            "|---|---|---|---|---|---|",
        ]
        for s in self.scenarios:
            lines.append(
                f"| {s.name} | {', '.join(s.tags)} | {s.cues} | {s.recall:.4f} "
                f"| {s.mrr:.4f} | {s.violations} |"
            )
        lines += ["", "## By tag", "", "| tag | scenarios | recall | mrr | violations |", "|---|---|---|---|---|"]
        for t in self.by_tag():
            lines.append(
                f"| {t.tag} | {t.scenarios} | {t.recall:.4f} | {t.mrr:.4f} | {t.violations} |"
            )
        lines.append("")
        return "\n".join(lines)
