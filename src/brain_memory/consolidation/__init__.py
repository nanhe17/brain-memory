"""Consolidation: replay + pattern extraction → semantic memories.

Deterministic grouping over the entity/topic index (never embedding
clustering), an honest statistical proposal when no LLM is configured, an
LLM proposal with a support-count quality gate when one is, structural
concept-keyed upserts with a full version trail, and incremental state so
only groups with new evidence are reprocessed.
"""

from brain_memory.consolidation.consolidator import Consolidator
from brain_memory.consolidation.grouping import CandidateGroup, candidate_groups
from brain_memory.consolidation.heuristic import HeuristicConsolidator
from brain_memory.consolidation.llm import LLMConsolidator
from brain_memory.consolidation.semantic_store import SemanticStore

__all__ = [
    "Consolidator",
    "CandidateGroup",
    "candidate_groups",
    "HeuristicConsolidator",
    "LLMConsolidator",
    "SemanticStore",
]
