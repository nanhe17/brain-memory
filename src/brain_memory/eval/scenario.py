"""Scenario schema, loading, and execution.

A scenario is a self-contained retrieval test: encode these episodes at these
relative ages, fire these cues, and check which episode *indexes* (0-based,
within the scenario) surface or must not surface.  Fresh in-memory engine per
run keeps scenarios independent and ids deterministic (index i -> id i+1).
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
    text: str = Field(min_length=1)
    source: str = "conversation"
    context: str | None = None
    days_ago: float = Field(default=0.0, ge=0.0)


class ScenarioCue(BaseModel):
    text: str = Field(min_length=1)
    expect: list[int] = Field(default_factory=list)
    forbid: list[int] = Field(default_factory=list)
    # semantic memories asserted by concept name (requires consolidate: true
    # or a semantic memory created by an earlier cue run)
    expect_concepts: list[str] = Field(default_factory=list)
    # assert the version of every expected semantic memory (reconsolidation
    # checks); recorded as semantic_version_ok, shown in cue misses
    expect_version: int | None = Field(default=None, ge=1)
    k: int | None = Field(default=None, ge=1)

    @field_validator("expect", "forbid")
    @classmethod
    def _non_negative(cls, value: list[int]) -> list[int]:
        if any(i < 0 for i in value):
            raise ValueError("episode indexes must be >= 0")
        return value

    @model_validator(mode="after")
    def _expects_something(self) -> "ScenarioCue":
        if not self.expect and not self.expect_concepts:
            raise ValueError("cue must set expect (episode indexes) or expect_concepts")
        return self


class ScenarioFile(BaseModel):
    name: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    episodes: list[ScenarioEpisode] = Field(min_length=1)
    cues: list[ScenarioCue] = Field(min_length=1)
    # run engine.consolidate() after encoding (tests the replay pipeline)
    consolidate: bool = False
    # encoded after the first consolidation, then consolidated again —
    # the reconsolidation round (Phase 4)
    later_episodes: list[ScenarioEpisode] = Field(default_factory=list)
    # run engine.decay() before the cues (tests the forgetting pipeline)
    decay: bool = False

    @model_validator(mode="after")
    def _indexes_in_range(self) -> "ScenarioFile":
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
    """Load and validate every ``*.yaml`` / ``*.yml`` scenario, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"scenario directory not found: {directory}")
    scenarios: list[ScenarioFile] = []
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenarios.append(ScenarioFile.model_validate(data))
    return scenarios


class CueResult(BaseModel):
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
    name: str
    tags: list[str]
    cues: list[CueResult]

    @property
    def recall(self) -> float:
        return sum(c.recall for c in self.cues) / len(self.cues)

    @property
    def mrr(self) -> float:
        return sum(c.mrr for c in self.cues) / len(self.cues)

    @property
    def violations(self) -> int:
        return sum(c.violations for c in self.cues)


class ScenarioRunner:
    """Runs one scenario against a fresh in-memory engine."""

    def __init__(self, config: MemoryConfig | None = None, default_k: int = 5) -> None:
        self._base_config = config
        self._default_k = default_k

    def _build_engine(self) -> MemoryEngine:
        base = self._base_config or MemoryConfig.from_env()
        run_config = base.model_copy(update={"db_path": ":memory:"})
        return MemoryEngine(run_config)

    def run(self, scenario: ScenarioFile) -> ScenarioResult:
        engine = self._build_engine()
        try:
            now = datetime.now(timezone.utc)
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
