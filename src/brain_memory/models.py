"""记忆引擎的核心数据模型。

这些模型构成系统的公开契约。其中最重要的是 :class:`ExtractedExperience`：
每条记忆都经由它进入存储，其字段（实体/话题/关键事实/强调信号）是下游
全部机制——检索因子、巩固分组、模式分离——的消费对象，务必保持稳定。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """当前 UTC 时间（时区感知）。"""
    return datetime.now(timezone.utc)


class MemoryStatus(str, Enum):
    """记忆的生命周期状态。

    Phase 1 只使用 ACTIVE 和 ARCHIVED（软删除）；FORGOTTEN 是衰减/遗忘
    阶段规划的终态——枚举值从第一天就定义好，让状态迁移始终显式。
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class SemanticKind(str, Enum):
    """语义记忆陈述的知识类型。

    CO_OCCURRENCE（共现）是确定性路径唯一会诚实输出的类型（"X 在 N 条
    记忆中反复出现"）；FACT / PREFERENCE / SCHEMA / GENERALIZATION 只有
    LLM 巩固器才有资格判定，启发式不假装理解语义。
    """

    FACT = "fact"
    PREFERENCE = "preference"
    CO_OCCURRENCE = "co_occurrence"
    SCHEMA = "schema"
    GENERALIZATION = "generalization"


class ConflictKind(str, Enum):
    """新证据与既有信念的关系分类（设计文档 §16）。

    冲突是记忆演化的重要信号，而非错误。UNRESOLVED 表示编码时发现的
    挑战，尚待再巩固裁决。
    """

    CONTRADICTION = "contradiction"
    EVOLUTION = "evolution"
    CORRECTION = "correction"
    CONTEXT_CHANGE = "context_change"
    UNRESOLVED = "unresolved"


class ConflictStatus(str, Enum):
    """冲突记录的处理状态：open（待裁决）→ resolved / dismissed。"""

    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class MemoryConflict(BaseModel):
    """新证据与既有信念之间的一次冲突记录。

    按 Phase 6 图谱的边的形状设计：两端（语义记忆 + 触发 episode）与
    方向都已就位，将来直接升格为 `contradicts`/`updates` 边。
    """

    id: int
    semantic_id: int
    kind: ConflictKind = ConflictKind.UNRESOLVED
    status: ConflictStatus = ConflictStatus.OPEN
    old_version: int
    statement_before: str | None = None
    trigger_episode_id: int | None = None
    trigger_kind: str = "consolidation"  # encode | consolidation | manual（触发来源）
    detected_at: datetime
    resolution_version: int | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedExperience(BaseModel):
    """ExperienceParser 的结构化输出——整个系统的数据契约。"""

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
    """不可变的情景记忆（append-only：永不合并、永不被覆写）。"""

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
    # 默认不序列化；由存储层按需加载
    embedding: np.ndarray | None = Field(default=None, repr=False, exclude=True)

    @property
    def is_active(self) -> bool:
        """是否处于活跃状态（归档/遗忘的记忆不参与检索）。"""
        return self.status is MemoryStatus.ACTIVE


class FactorScores(BaseModel):
    """多因子检索的逐因子得分，每个因子已归一化到 [0, 1]。"""

    semantic: float = 0.0
    keyword: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    frequency: float = 0.0
    entity: float = 0.0
    context: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """导出为普通字典（供 API 序列化）。"""
        return self.model_dump()


class SemanticMemory(BaseModel):
    """由多条情景记忆巩固而来的知识。

    身份是结构化的：每个 ``concept``（归一化的组键）只有一行，绝不因
    文本相似而重复。每次状态变更都会追加一行 :class:`MemoryVersion`，
    因此"你为什么相信这个"永远可以从证据 + 历史中找到答案。
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
    """语义记忆的一个历史状态（完整版本链）。"""

    id: int
    semantic_id: int
    version: int
    statement: str
    confidence: float
    evidence_ids: list[int] = Field(default_factory=list)
    created_at: datetime
    change_reason: str


class PatternProposal(BaseModel):
    """巩固器对某个概念组给出的候选知识陈述。

    ``change_kind`` 是针对既有陈述的再巩固裁决（"consistent" = 普通精炼）；
    ``metadata`` 携带确定性信号，如偏好极性统计。
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
    """图节点类型：两个实例节点 + 一个概念伪节点。"""

    EPISODE = "episode"
    SEMANTIC = "semantic"
    CONCEPT = "concept"  # 实体/话题伪节点——图的连接组织


class EdgeKind(str, Enum):
    """图边类型（每类的来源见表内注释）。"""

    MENTIONS = "mentions"                    # episode -> 概念（tags）
    DERIVED_FROM = "derived_from"            # semantic -> episode（证据）
    CONTRADICTS = "contradicts"              # semantic -> episode（冲突记录）
    CO_OCCURS_WITH = "co_occurs_with"        # 概念 <-> 概念（共享 episode）
    SIMILAR_TO = "similar_to"                # episode <-> episode（向量 kNN）
    RELATED_TO = "related_to"                # 显式（memory_links）
    CAUSED_BY = "caused_by"                  # 显式（memory_links）
    PART_OF = "part_of"                      # 显式（memory_links）


EXPLICIT_RELATIONS = ("related_to", "caused_by", "similar_to", "part_of")


class GraphNode(BaseModel):
    """图节点：ref 是统一引用（"e12" / "s3" / "c:java"）。"""

    ref: str
    kind: NodeKind
    id: int | None = None  # 概念节点没有实例 id
    label: str
    status: MemoryStatus | None = None
    detail: str = ""  # 陈述或截断的内容


class GraphEdge(BaseModel):
    """带来源的类型化图边（provenance 标注这条边从哪张表推导而来）。"""

    source: str
    target: str
    kind: EdgeKind
    weight: float = 1.0
    provenance: str  # tags | evidence | conflict | co_occurrence | vector | explicit


class GraphSubgraph(BaseModel):
    """一次邻域查询的结果：中心 + 节点集 + 边集。"""

    center: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class MemoryLink(BaseModel):
    """显式断言的关系（agent 或用户），连接两个实例节点。

    可推导的关系（mentions/证据/冲突）不存这里——它们是各自表上的
    读模型。
    """

    id: int
    source_kind: NodeKind
    source_id: int
    target_kind: NodeKind
    target_id: int
    relation: str  # 封闭词表：related_to | caused_by | similar_to | part_of
    weight: float = 1.0
    created_by: str = "agent"
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResult(BaseModel):
    """一条被召回的记忆，以及"为什么召回它"的解释。

    ``kind="semantic"`` 的命中在 ``semantic`` 中携带真正的
    :class:`SemanticMemory`，同时在 ``episode`` 中携带一个只读*视图*
    （content=statement、created_at=updated_at、importance=confidence、
    entities=[concept]），让所有消费方（重排器/prompt 块/API）统一处理。
    视图的 id 属于语义 id 空间——绝不要把它回传给 episode 接口。
    """

    episode: Episode
    kind: Literal["episodic", "semantic"] = "episodic"
    semantic: SemanticMemory | None = None
    score: float
    factors: FactorScores
    reasons: list[str] = Field(default_factory=list)
    # 盲评 LLM 重排器对最终得分有贡献时设置
    llm_relevance: float | None = None
    # 经图扩展进入结果（而非直接匹配）时设置
    expanded: bool = False

    @property
    def content(self) -> str:
        """命中的文本内容（情景=原文，语义=陈述）。"""
        return self.episode.content

    @property
    def memory_id(self) -> int:
        """记忆 id（语义命中时是语义 id）。"""
        return self.episode.id

    @property
    def is_semantic(self) -> bool:
        """是否为语义（巩固知识）命中。"""
        return self.kind == "semantic"


class EncodeResult(BaseModel):
    """编码结果：新episode 或命中的重复项。"""

    episode: Episode
    duplicate: bool = False
    # 本条 episode 的纠正信号挑战了某个既有语义记忆时设置（再巩固入口，文档 §15）
    challenge: "MemoryConflict | None" = None


class WorkingMemoryState(BaseModel):
    """工作记忆快照（当前任务状态，不是一个存储）。"""

    current_goal: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    recent_episode_ids: list[int] = Field(default_factory=list)
    retrieved_memory_ids: list[int] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    token_budget: int = 2000


class EngineStats(BaseModel):
    """引擎整体统计。"""

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
    """一次 ``consolidate()`` 的结果。"""

    groups_considered: int = 0
    created: list[SemanticMemory] = Field(default_factory=list)
    updated: list[SemanticMemory] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # "group_key: 原因"
    conflicts: list[MemoryConflict] = Field(default_factory=list)

    @property
    def touched(self) -> list[SemanticMemory]:
        """本次新建 + 更新的语义记忆。"""
        return [*self.created, *self.updated]


class DecayReport(BaseModel):
    """一次 ``decay()`` 扫描的结果。

    所有迁移都是软的：归档可恢复，遗忘亦可恢复——本阶段绝不物理删除。
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
        """本次发生的状态迁移总数。"""
        return (
            len(self.archived_episode_ids)
            + len(self.forgotten_episode_ids)
            + len(self.archived_semantic_ids)
            + len(self.forgotten_semantic_ids)
        )
