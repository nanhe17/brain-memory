"""经验解析：把原始文本转换为 :class:`ExtractedExperience`。

解析器契约（``base.py``）是系统的数据边界——启发式与 LLM 两种解析器
都实现它，引擎的其余部分永远不需要知道一条记忆由谁产出。
"""

from brain_memory.extraction.base import ExperienceParser, ExtractedExperience
from brain_memory.extraction.heuristic import HeuristicExperienceParser

__all__ = ["ExperienceParser", "ExtractedExperience", "HeuristicExperienceParser"]
