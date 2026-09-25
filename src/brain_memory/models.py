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


class ConflictKind(str, Enum):
    """How new evidence relates to an existing semantic memory (doc §16).

    Conflicts are signals of memory evolution, not errors.  UNRESOLVED marks
    an encode-time challenge that reconsolidation has not classified yet.
    """

    CONTRADICTION = "contradiction"
    EVOLUTION = "evolution"
    CORRECTION = "correction"
    CONTEXT_CHANGE = "context_change"
    UNRESOLVED = "unresolved"


class ConflictStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class MemoryConflict(BaseModel):
    """A recorded clash between new evidence and an existing belief.

    Designed to become Memory-Graph edges (Phase 6): both endpoints (semantic
    memory + triggering episode) and the direction are already here.
    """

    id: int
    semantic_id: int
    kind: ConflictKind = ConflictKind.UNRESOLVED
    status: ConflictStatus = ConflictStatus.OPEN
    old_version: int
    statement_before: str | None = None
    trigger_episode_id: int | None = None
    trigger_kind: str = "consolidation"  # encode | consolidation | manual
    detected_at: datetime
    resolution_version: int | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """A consolidator's candidate knowledge statement for one concept group.

    ``change_kind`` is the reconsolidation verdict against the existing
    statement ("consistent" = plain refinement); ``metadata`` carries
    deterministic signals such as preference-polarity counts.
    """

    concept: str
    statement: str
    kind: SemanticKind = SemanticKind.CO_OCCURRENCE
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_indexes: list[int] = Field(default_factory=list)
    change_kind: Literal[
        "consistent", "contradiction", "evolution", "correction", "context_change"
    ] = "consistent"
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeKind(str, Enum):
    EPISODE = "episode"
    SEMANTIC = "semantic"
    CONCEPT = "concept"  # entity/topic pseudo-node — the graph's connective tissue


class EdgeKind(str, Enum):
    MENTIONS = "mentions"                    # episode -> concept (tags)
    DERIVED_FROM = "derived_from"            # semantic -> episode (evidence)
    CONTRADICTS = "contradicts"              # semantic -> episode (conflict record)
    CO_OCCURS_WITH = "co_occurs_with"        # concept <-> concept (shared episodes)
    SIMILAR_TO = "similar_to"                # episode <-> episode (vector kNN)
    RELATED_TO = "related_to"                # explicit (memory_links)
    CAUSED_BY = "caused_by"                  # explicit (memory_links)
    PART_OF = "part_of"                      # explicit (memory_links)


EXPLICIT_RELATIONS = ("related_to", "caused_by", "similar_to", "part_of")


class GraphNode(BaseModel):
    ref: str  # "e12" | "s3" | "c:java"
    kind: NodeKind
    id: int | None = None  # None for concept nodes
    label: str
    status: MemoryStatus | None = None
    detail: str = ""  # statement / truncated content


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: EdgeKind
    weight: float = 1.0
    provenance: str  # tags | evidence | conflict | co_occurrence | vector | explicit


class GraphSubgraph(BaseModel):
    center: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class MemoryLink(BaseModel):
    """An explicitly asserted relationship (agent or user) between two
    instance nodes.  Derived relations (mentions/evidence/conflicts) are NOT
    stored here — they are read models over their own tables."""

    id: int
    source_kind: NodeKind
    source_id: int
    target_kind: NodeKind
    target_id: int
    relation: str  # closed vocabulary: related_to | caused_by | similar_to | part_of
    weight: float = 1.0
    created_by: str = "agent"
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    # Set when this hit entered via graph expansion rather than direct matching.
    expanded: bool = False

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
    # Set when this episode's correction signals challenge an existing
    # semantic memory (reconsolidation entry, doc §15).
    challenge: "MemoryConflict | None" = None


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
    open_conflicts: int = 0


class ConsolidationReport(BaseModel):
    """Outcome of one ``consolidate()`` run."""

    groups_considered: int = 0
    created: list[SemanticMemory] = Field(default_factory=list)
    updated: list[SemanticMemory] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # "group_key: reason"
    conflicts: list[MemoryConflict] = Field(default_factory=list)

    @property
    def touched(self) -> list[SemanticMemory]:
        return [*self.created, *self.updated]


class DecayReport(BaseModel):
    """Outcome of one ``decay()`` sweep.

    All transitions are soft: archived memories can be restored, forgotten
    ones too — nothing is ever physically deleted in this phase.
    """

    swept_episodes: int = 0
    swept_semantics: int = 0
    archived_episode_ids: list[int] = Field(default_factory=list)
    forgotten_episode_ids: list[int] = Field(default_factory=list)
    archived_semantic_ids: list[int] = Field(default_factory=list)
    forgotten_semantic_ids: list[int] = Field(default_factory=list)
    protected_evidence_count: int = 0
    dry_run: bool = False

    @property
    def transitions(self) -> int:
        return (
            len(self.archived_episode_ids)
            + len(self.forgotten_episode_ids)
            + len(self.archived_semantic_ids)
            + len(self.forgotten_semantic_ids)
        )
