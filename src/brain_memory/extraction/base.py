"""Parser contract — the extraction boundary of the whole system.

Everything downstream (retrieval factors, Phase-3 consolidation grouping,
pattern separation) consumes the structured fields produced here.  Changing
this schema changes the system's contract; treat it accordingly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from brain_memory.models import ExtractedExperience


@runtime_checkable
class ExperienceParser(Protocol):
    """Turns raw experience text into a structured :class:`ExtractedExperience`."""

    def parse(
        self,
        text: str,
        *,
        source: str = "conversation",
        context: str | None = None,
        timestamp: datetime | None = None,
    ) -> ExtractedExperience:
        """Parse *text*; implementations must be side-effect free.

        ``timestamp=None`` means "now" (UTC).  Implementations should never
        raise on unusual input — degrade gracefully instead (a memory with
        weak metadata is better than a lost experience).
        """
        ...
