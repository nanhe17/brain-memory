"""Core data models for the memory engine.

These models form the public contract of the system.  The most important one
is :class:`ExtractedExperience`: every episode enters the store through it, so
its fields (entities / topics / key_facts / emphasis signals) are what all
downstream machinery — retrieval factors, future consolidation, pattern
separation — consumes.  Keep it stable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStatus(str, Enum):
    """Lifecycle state of a memory.

    Phase 1 only uses ACTIVE and ARCHIVED (soft delete).  FORGOTTEN is the
    terminal state planned for the decay/forgetting phase — the value exists
    now so transitions are explicit from the start.
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class SemanticKind(str, Enum):
    """What kind of knowledge a semantic memory states.

    CO_OCCURRENCE is the deterministic path's honest output ("X shows up in N
    memories"); FACT / PREFERENCE / SCHEMA / GENERALIZATION are only produced
    by the LLM consolidator, which can actually judge meaning.
    """

    FACT = "fact"
    PREFERENCE = "preference"
    CO_OCCURRENCE = "co_occurrence"
    SCHEMA = "schema"
    GENERALIZATION = "generalization"


class ExtractedExperience(BaseModel):
    """Structured output of an ExperienceParser — the system's data contract."""

    content: str
    entities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    emphasis_signals: list[str] = Field(default_factory=list)
    context: str | None = None
    source: str = "conversation"
    importance: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utcnow)


class Episode(BaseModel):
    """An immutable episodic memory (append-only; never merged or overwritten)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    content: str
    content_hash: str
    entities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    emphasis_signals: list[str] = Field(default_factory=list)
    context: str | None = None
    source: str = "conversation"
    created_at: datetime
    importance: float
    confidence: float
    access_count: int = 0
    last_accessed: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Not persisted in the default dump; loaded on demand by the store.
    embedding: np.ndarray | None = Field(default=None, repr=False, exclude=True)

    @property
    def is_active(self) -> bool:
        return self.status is MemoryStatus.ACTIVE


class FactorScores(BaseModel):
    """Per-factor retrieval scores, each normalized to [0, 1]."""

    semantic: float = 0.0
    keyword: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    frequency: float = 0.0
    entity: float = 0.0
    context: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class SemanticMemory(BaseModel):
    """Consolidated knowledge distilled from multiple episodes.

    Identity is structural: one row per ``concept`` (the normalized group key),
    never duplicated by textual similarity.  Every state change appends a
    :class:`MemoryVersion` row, so "why do you believe this" is always
    answerable from evidence + history.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    concept: str
    kind: SemanticKind = SemanticKind.CO_OCCURRENCE
    statement: str
    confidence: float
    evidence_ids: list[int] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    version: int = 1
    status: MemoryStatus = MemoryStatus.ACTIVE
    access_count: int = 0
    last_accessed: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: np.ndarray | None = Field(default=None, repr=False, exclude=True)


class MemoryVersion(BaseModel):
    """One historical state of a semantic memory (full version trail)."""

    id: int
    semantic_id: int
    version: int
    statement: str
    confidence: float
    evidence_ids: list[int] = Field(default_factory=list)
    created_at: datetime
    change_reason: str


class PatternProposal(BaseModel):
    """A consolidator's candidate knowledge statement for one concept group."""

    concept: str
    statement: str
    kind: SemanticKind = SemanticKind.CO_OCCURRENCE
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_indexes: list[int] = Field(default_factory=list)


class RecallResult(BaseModel):
    """A retrieved memory plus an explanation of why it was recalled.

    ``kind="semantic"`` hits carry the :class:`SemanticMemory` in ``semantic``
    and a read-only *view* of it in ``episode`` (content=statement,
    created_at=updated_at, importance=confidence, entities=[concept]) so that
    consumers (reranker, prompt block, API) can treat every hit uniformly.
    The view's id lives in the semantic id space — never feed it back into
    episode APIs.
    """

    episode: Episode
    kind: Literal["episodic", "semantic"] = "episodic"
    semantic: SemanticMemory | None = None
    score: float
    factors: FactorScores
    reasons: list[str] = Field(default_factory=list)
    # Set when the blind LLM reranker contributed to the final score.
    llm_relevance: float | None = None

    @property
    def content(self) -> str:
        return self.episode.content

    @property
    def memory_id(self) -> int:
        return self.episode.id

    @property
    def is_semantic(self) -> bool:
        return self.kind == "semantic"


class EncodeResult(BaseModel):
    episode: Episode
    duplicate: bool = False


class WorkingMemoryState(BaseModel):
    """Snapshot of the agent's working memory (current-task state, not a store)."""

    current_goal: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    recent_episode_ids: list[int] = Field(default_factory=list)
    retrieved_memory_ids: list[int] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    token_budget: int = 2000


class EngineStats(BaseModel):
    total_episodes: int
    active: int
    archived: int
    forgotten: int
    distinct_entities: int
    avg_importance: float
    oldest_created_at: datetime | None = None
    newest_created_at: datetime | None = None
    semantic_memories: int = 0


class ConsolidationReport(BaseModel):
    """Outcome of one ``consolidate()`` run."""

    groups_considered: int = 0
    created: list[SemanticMemory] = Field(default_factory=list)
    updated: list[SemanticMemory] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # "group_key: reason"

    @property
    def touched(self) -> list[SemanticMemory]:
        return [*self.created, *self.updated]
