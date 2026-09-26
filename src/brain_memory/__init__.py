"""Brain-inspired Memory Engine——面向 AI agent 的类脑长期记忆架构（Phase 1 MVP 起步，
现已是七阶段完整实现）。

快速上手::

    from brain_memory import MemoryEngine

    engine = MemoryEngine()          # 从 MEMORY_* 环境变量读配置
    engine.encode("User is building a P-51 Minecraft mod")
    hits = engine.recall("the airplane mod")
    print(hits[0].episode.content, hits[0].reasons)
"""

from brain_memory.config import MemoryConfig, RetrievalWeights
from brain_memory.engine import MemoryEngine
from brain_memory.models import (
    ConflictKind,
    ConflictStatus,
    ConsolidationReport,
    DecayReport,
    EncodeResult,
    EngineStats,
    Episode,
    ExtractedExperience,
    FactorScores,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    MemoryConflict,
    MemoryLink,
    MemoryStatus,
    MemoryVersion,
    NodeKind,
    PatternProposal,
    RecallResult,
    SemanticKind,
    SemanticMemory,
    WorkingMemoryState,
)
from brain_memory.working.working_memory import WorkingMemory

__version__ = "0.1.0"

__all__ = [
    "MemoryEngine",
    "MemoryConfig",
    "RetrievalWeights",
    "WorkingMemory",
    "ConflictKind",
    "ConflictStatus",
    "MemoryConflict",
    "ConsolidationReport",
    "DecayReport",
    "EncodeResult",
    "EngineStats",
    "Episode",
    "ExtractedExperience",
    "FactorScores",
    "GraphEdge",
    "GraphNode",
    "GraphSubgraph",
    "MemoryLink",
    "MemoryStatus",
    "MemoryVersion",
    "NodeKind",
    "PatternProposal",
    "RecallResult",
    "SemanticKind",
    "SemanticMemory",
    "WorkingMemoryState",
    "__version__",
]
