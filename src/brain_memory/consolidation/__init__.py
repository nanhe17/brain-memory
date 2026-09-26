"""巩固：重放 + 模式提取 → 语义记忆。

基于实体/话题索引的确定性分组（绝不使用 embedding 聚类）、无 LLM 时
诚实的统计型提案、有 LLM 时带支持数质量门的 LLM 提案、结构化概念键
upsert 与完整版本链，以及"只有新证据的组才会被重处理"的增量状态。
"""

from brain_memory.consolidation.consolidator import Consolidator
from brain_memory.consolidation.conflicts import ConflictStore
from brain_memory.consolidation.grouping import CandidateGroup, candidate_groups
from brain_memory.consolidation.heuristic import HeuristicConsolidator
from brain_memory.consolidation.llm import LLMConsolidator
from brain_memory.consolidation.semantic_store import SemanticStore

__all__ = [
    "Consolidator",
    "ConflictStore",
    "CandidateGroup",
    "candidate_groups",
    "HeuristicConsolidator",
    "LLMConsolidator",
    "SemanticStore",
]
