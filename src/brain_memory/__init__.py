"""Brain-inspired Memory Engine — Phase 1 MVP.

Quick start::

    from brain_memory import MemoryEngine

    engine = MemoryEngine()          # config from MEMORY_* env vars
    engine.encode("User is building a P-51 Minecraft mod")
    hits = engine.recall("the airplane mod")
    print(hits[0].episode.content, hits[0].reasons)
"""

from brain_memory.config import MemoryConfig, RetrievalWeights
from brain_memory.engine import MemoryEngine
from brain_memory.models import (
    ConsolidationReport,
    EncodeResult,
    EngineStats,
    Episode,
    ExtractedExperience,
    FactorScores,
    MemoryStatus,
    MemoryVersion,
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
    "ConsolidationReport",
    "EncodeResult",
    "EngineStats",
    "Episode",
    "ExtractedExperience",
    "FactorScores",
    "MemoryStatus",
    "MemoryVersion",
    "PatternProposal",
    "RecallResult",
    "SemanticKind",
    "SemanticMemory",
    "WorkingMemoryState",
    "__version__",
]
