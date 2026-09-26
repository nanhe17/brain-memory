"""场景 schema、加载与执行。

一个场景即一个自包含的检索测试：按相对时间编码这些 episode，触发
这些 cue，检查哪些 episode *索引*（场景内 0 基）必须出现或不得出现。
每次运行用全新的内存引擎，保持场景独立且 id 确定（索引 i -> id i+1）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from brain_memory.config import MemoryConfig
from brain_memory.engine import MemoryEngine
from brain_memory.eval.metrics import mrr, precision_at_k, recall_at_k, violation_count


class ScenarioEpisode(BaseModel):
    """场景中的一个待编码记忆。"""

    text: str = Field(min_length=1)
    source: str = "conversation"
    context: str | None = None
    days_ago: float = Field(default=0.0, ge=0.0)


class ScenarioCue(BaseModel):
    """一个检索 cue 及其断言。"""

    text: str = Field(min_length=1)
    expect: list[int] = Field(default_factory=list)
    forbid: list[int] = Field(default_factory=list)
    # 按概念名断言语义记忆（需要 consolidate: true，或之前的运行已产生）
    expect_concepts: list[str] = Field(default_factory=list)
    # 断言期望语义记忆的版本号（再巩固检查）；失败记入 semantic_version_ok
    expect_version: int | None = Field(default=None, ge=1)
    k: int | None = Field(default=None, ge=1)

    @field_validator("expect", "forbid")
    @classmethod
    def _non_negative(cls, value: list[int]) -> list[int]:
        """episode 索引必须非负。"""
        if any(i < 0 for i in value):
            raise ValueError("episode indexes must be >= 0")
        return value

    @model_validator(mode="after")
    def _expects_something(self) -> "ScenarioCue":
        """cue 至少要断言一件事，否则没有意义。"""
        if not self.expect and not self.expect_concepts:
            raise ValueError("cue must set expect (episode indexes) or expect_concepts")
        return self


class ScenarioFile(BaseModel):
    """一个完整的评测场景。"""

    name: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    episodes: list[ScenarioEpisode] = Field(min_length=1)
    cues: list[ScenarioCue] = Field(min_length=1)
    # 编码后运行 engine.consolidate()（测试巩固管线）
    consolidate: bool = False
    # 首轮巩固后再编码，然后再巩固一轮——再巩固回合（Phase 4）
    later_episodes: list[ScenarioEpisode] = Field(default_factory=list)
    # cue 之前运行 engine.decay()（测试遗忘管线）
    decay: bool = False

    @model_validator(mode="after")
    def _indexes_in_range(self) -> "ScenarioFile":
        """expect/forbid 的索引必须落在 episode 范围内。"""
        limit = len(self.episodes)
        for cue in self.cues:
            for field_name, indexes in (("expect", cue.expect), ("forbid", cue.forbid)):
                bad = [i for i in indexes if i >= limit]
                if bad:
                    raise ValueError(
                        f"cue '{cue.text[:30]}…' {field_name} indexes {bad} out of range "
                        f"(scenario has {limit} episodes)"
                    )
        return self


def load_scenarios(directory: str | Path) -> list[ScenarioFile]:
    """加载并校验目录下全部 ``*.yaml`` / ``*.yml`` 场景（按名称排序）。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"scenario directory not found: {directory}")
    scenarios: list[ScenarioFile] = []
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenarios.append(ScenarioFile.model_validate(data))
    return scenarios


class CueResult(BaseModel):
    """单个 cue 的评测结果。"""

    cue: str
    k: int
    ranked_ids: list[str]
    expected_ids: list[str]
    forbidden_ids: list[str]
    recall: float
    mrr: float
    precision: float
    violations: int
    semantic_version_ok: bool | None = None


class ScenarioResult(BaseModel):
    """一个场景的评测结果（聚合所有 cue）。"""

    name: str
    tags: list[str]
    cues: list[CueResult]

    @property
    def recall(self) -> float:
        """cue 级 recall 的均值。"""
        return sum(c.recall for c in self.cues) / len(self.cues)

    @property
    def mrr(self) -> float:
        """cue 级 MRR 的均值。"""
        return sum(c.mrr for c in self.cues) / len(self.cues)

    @property
    def violations(self) -> int:
        """全部禁用泄漏之和。"""
        return sum(c.violations for c in self.cues)


class ScenarioRunner:
    """在全新内存引擎上运行一个场景。"""

    def __init__(self, config: MemoryConfig | None = None, default_k: int = 5) -> None:
        self._base_config = config
        self._default_k = default_k

    def _build_engine(self) -> MemoryEngine:
        """构建运行引擎（db 强制 :memory:，其余继承配置）。"""
        base = self._base_config or MemoryConfig.from_env()
        run_config = base.model_copy(update={"db_path": ":memory:"})
        return MemoryEngine(run_config)

    def run(self, scenario: ScenarioFile) -> ScenarioResult:
        """执行场景：编码 → 可选巩固 → 可选补编/再巩固 → 可选衰减 → 逐 cue 评测。"""
        engine = self._build_engine()
        try:
            now = datetime.now(timezone.utc)
            # 编码基础 episode，记录 索引 -> id 映射
            id_by_index: dict[int, int] = {}
            for index, episode in enumerate(scenario.episodes):
                created_at = now - timedelta(days=episode.days_ago)
                result = engine.encode(
                    episode.text,
                    source=episode.source,
                    context=episode.context,
                    created_at=created_at,
                )
                id_by_index[index] = result.episode.id

            concept_to_memory: dict[str, object] = {}
            if scenario.consolidate:
                engine.consolidate(max_groups=25, max_episodes_per_group=50)
                concepts = {
                    concept for cue in scenario.cues for concept in cue.expect_concepts
                }
                for concept in concepts:
                    memory = engine.find_semantic(concept)
                    if memory is not None:
                        concept_to_memory[concept] = memory

            # 再巩固回合：补编 episode 后再巩固一次，刷新概念映射
            if scenario.later_episodes:
                for episode in scenario.later_episodes:
                    created_at = now - timedelta(days=episode.days_ago)
                    engine.encode(
                        episode.text,
                        source=episode.source,
                        context=episode.context,
                        created_at=created_at,
                    )
                if scenario.consolidate:
                    engine.consolidate(max_groups=25, max_episodes_per_group=50)
                    for concept in list(concept_to_memory):
                        memory = engine.find_semantic(concept)
                        if memory is not None:
                            concept_to_memory[concept] = memory

            if scenario.decay:
                engine.decay()

            cue_results: list[CueResult] = []
            for cue in scenario.cues:
                k = cue.k or self._default_k
                hits = engine.recall(cue.text, k=k)
                # 排名与期望统一到 "e{id}"/"s{id}" 标签空间
                ranked = [
                    f"s{hit.semantic.id}" if hit.is_semantic else f"e{hit.episode.id}"
                    for hit in hits
                ]
                expected = {f"e{id_by_index[i]}" for i in cue.expect}
                expected |= {
                    f"s{concept_to_memory[concept].id}"
                    for concept in cue.expect_concepts
                    if concept in concept_to_memory
                }
                forbidden = {f"e{id_by_index[i]}" for i in cue.forbid}

                # 语义版本断言（如再巩固场景断言 v2）
                version_ok: bool | None = None
                if cue.expect_version is not None:
                    version_ok = bool(concept_to_memory) and all(
                        getattr(concept_to_memory[c], "version", None) == cue.expect_version
                        for c in cue.expect_concepts
                        if c in concept_to_memory
                    )
                cue_results.append(
                    CueResult(
                        cue=cue.text,
                        k=k,
                        ranked_ids=ranked,
                        expected_ids=sorted(expected),
                        forbidden_ids=sorted(forbidden),
                        recall=recall_at_k(expected, ranked, k),
                        mrr=mrr(expected, ranked),
                        precision=precision_at_k(expected, ranked, k),
                        violations=violation_count(forbidden, ranked, k),
                        semantic_version_ok=version_ok,
                    )
                )
            return ScenarioResult(name=scenario.name, tags=scenario.tags, cues=cue_results)
        finally:
            engine.close()
