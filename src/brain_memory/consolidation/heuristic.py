"""Deterministic proposal: honest co-occurrence statements, nothing more.

Without an LLM the system cannot judge *meaning*, so it only states what it
can count: that a concept recurs across N memories and which topics those
memories touch.  Confidence is hard-capped at 0.55 — deliberately below
anything the LLM path can produce — and the kind is always ``co_occurrence``.
Preference/fact-level knowledge requires the LLM consolidator.
"""

from __future__ import annotations

from collections import Counter

from brain_memory.models import Episode, PatternProposal, SemanticKind

_CJK_RE = None  # compiled lazily in _cjk_ratio


def _cjk_ratio(text: str) -> float:
    import re

    global _CJK_RE
    if _CJK_RE is None:
        _CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / max(len(text), 1)


class HeuristicConsolidator:
    """Statistical proposals only; see module docstring."""

    def propose(
        self,
        concept: str,
        episodes: list[Episode],
        *,
        min_support: int,
    ) -> PatternProposal | None:
        supporting_indexes = [
            index
            for index, episode in enumerate(episodes)
            if self._supports(concept, episode)
        ]
        if len(supporting_indexes) < min_support:
            return None

        supporting = [episodes[index] for index in supporting_indexes]
        topics = self._top_topics(concept, supporting)
        confidence = round(min(0.55, 0.20 + 0.08 * len(supporting)), 3)

        if _cjk_ratio(" ".join(ep.content for ep in supporting)) >= 0.3:
            statement = self._statement_zh(concept, len(supporting), topics)
        else:
            statement = self._statement_en(concept, len(supporting), topics)

        return PatternProposal(
            concept=concept,
            statement=statement,
            kind=SemanticKind.CO_OCCURRENCE,
            confidence=confidence,
            supporting_indexes=supporting_indexes,
        )

    @staticmethod
    def _supports(concept: str, episode: Episode) -> bool:
        tags = {t.casefold() for t in [*episode.entities, *episode.topics]}
        return concept.casefold() in tags or concept.casefold() in episode.content.casefold()

    @staticmethod
    def _top_topics(concept: str, episodes: list[Episode], limit: int = 4) -> list[str]:
        counter: Counter[str] = Counter()
        for episode in episodes:
            for tag in {t.casefold() for t in [*episode.entities, *episode.topics]}:
                if tag != concept.casefold():
                    counter[tag] += 1
        # frequency first, then deterministic name order
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        return [topic for topic, _count in ranked[:limit]]

    @staticmethod
    def _statement_zh(concept: str, count: int, topics: list[str]) -> str:
        head = f"{concept} 在 {count} 条记忆中反复出现"
        if topics:
            return f"{head}，主要涉及 {'、'.join(topics)}"
        return head

    @staticmethod
    def _statement_en(concept: str, count: int, topics: list[str]) -> str:
        head = f"{concept} shows up across {count} memories"
        if topics:
            return f"{head}, mainly involving {', '.join(topics)}"
        return head
