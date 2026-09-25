"""Experience parsing: turning raw text into :class:`ExtractedExperience`.

The parser contract (``base.py``) is the system's data boundary — heuristic
and LLM-backed parsers both implement it, so the rest of the engine never
knows which one produced a memory.
"""

from brain_memory.extraction.base import ExperienceParser, ExtractedExperience
from brain_memory.extraction.heuristic import HeuristicExperienceParser

__all__ = ["ExperienceParser", "ExtractedExperience", "HeuristicExperienceParser"]
