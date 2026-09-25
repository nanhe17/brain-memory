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
from typing import Any

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


class RecallResult(BaseModel):
    """A retrieved memory plus an explanation of why it was recalled."""

    episode: Episode
    score: float
    factors: FactorScores
    reasons: list[str] = Field(default_factory=list)
    # Set when the blind LLM reranker contributed to the final score.
    llm_relevance: float | None = None


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
