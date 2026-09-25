"""Deterministic proposal: honest co-occurrence statements, nothing more.

Without an LLM the system cannot judge *meaning*, so it only states what it
can count: that a concept recurs across N memories and which topics those
memories touch, plus — for reconsolidation — which *direction* preference
markers point (positive/negative/mixed, from the parser's own marker
lexicons).  Confidence is hard-capped at 0.55 — deliberately below anything
the LLM path can produce — and the kind is always ``co_occurrence``.
Preference/fact-level knowledge requires the LLM consolidator.
"""

from __future__ import annotations

import re
from collections import Counter

from brain_memory.models import Episode, PatternProposal, SemanticKind, SemanticMemory

_POSITIVE_MARKERS = ("喜欢", "偏好", "偏爱", "热衷", "love", "like", "prefer", "enjoy")
_NEGATIVE_MARKERS = ("不喜欢", "讨厌", "反感", "不想", "hate", "dislike", "not a fan")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / max(len(text), 1)


def episode_polarity(content: str) -> str | None:
    """The preference direction of one episode: 'pos' | 'neg' | 'mixed' | None.

    Negative markers are scrubbed before the positive check so 「不喜欢」 is
    not double-counted as containing 「喜欢」.
    """
    low = content.lower()
    neg_hits = [m for m in _NEGATIVE_MARKERS if m in low]
    if neg_hits:
        scrubbed = low
        for marker in neg_hits:
            scrubbed = scrubbed.replace(marker, " ")
        return "mixed" if any(m in scrubbed for m in _POSITIVE_MARKERS) else "neg"
    return "pos" if any(m in low for m in _POSITIVE_MARKERS) else None


def polarity_counts(episodes: list[Episode]) -> dict[str, int]:
    counts = {"pos": 0, "neg": 0, "mixed": 0}
    for episode in episodes:
        sign = episode_polarity(episode.content)
        if sign:
            counts[sign] += 1
    return counts


def dominant_sign(counts: dict[str, int]) -> str | None:
    """'pos' | 'neg' | 'mixed' | None over a polarity tally."""
    if counts["pos"] > counts["neg"]:
        return "pos"
    if counts["neg"] > counts["pos"]:
        return "neg"
    if counts["pos"] + counts["neg"] > 0:
        return "mixed"
    return None


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

    def repropose(
        self,
        concept: str,
        episodes: list[Episode],
        existing: SemanticMemory,
        *,
        min_support: int,
    ) -> PatternProposal | None:
        """Reconsolidation for an existing memory: same honest counting, plus
        preference-polarity tracking.  A dominant-sign shift across versions
        (pos/neg/mixed) yields an evolution statement; a first polarity
        observation has no baseline and is always 'consistent'."""
        base = self.propose(concept, episodes, min_support=min_support)
        if base is None:
            return None

        supporting = [episodes[index] for index in base.supporting_indexes]
        polarity = polarity_counts(supporting)
        prev = (existing.metadata or {}).get("polarity")
        prev_sign = dominant_sign(prev) if isinstance(prev, dict) else None
        current_sign = dominant_sign(polarity)

        change_kind = "consistent"
        statement = base.statement
        if prev is not None and prev_sign and current_sign and prev_sign != current_sign:
            change_kind = "evolution"
            if _cjk_ratio(statement) >= 0.3 or _cjk_ratio(existing.statement) >= 0.3:
                statement = self._shift_statement_zh(concept, prev, polarity)
            else:
                statement = self._shift_statement_en(concept, prev, polarity)

        return base.model_copy(
            update={"statement": statement, "change_kind": change_kind,
                    "metadata": {"polarity": polarity}}
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
    def _sign_label_zh(sign: str) -> str:
        return {"pos": "正面", "neg": "负面", "mixed": "正负并存"}[sign]

    @staticmethod
    def _sign_label_en(sign: str) -> str:
        return {"pos": "positive", "neg": "negative", "mixed": "mixed"}[sign]

    @classmethod
    def _shift_statement_zh(cls, concept: str, prev: dict, curr: dict) -> str:
        return (
            f"{concept} 的态度信号出现变化：此前以{cls._sign_label_zh(dominant_sign(prev))}为主"
            f"（{prev.get('pos', 0)} 正 / {prev.get('neg', 0)} 负），"
            f"近期以{cls._sign_label_zh(dominant_sign(curr))}为主"
            f"（{curr.get('pos', 0)} 正 / {curr.get('neg', 0)} 负）"
        )

    @classmethod
    def _shift_statement_en(cls, concept: str, prev: dict, curr: dict) -> str:
        return (
            f"{concept} attitude signals shifted: earlier mostly "
            f"{cls._sign_label_en(dominant_sign(prev))} ({prev.get('pos', 0)} pos / "
            f"{prev.get('neg', 0)} neg), recently mostly "
            f"{cls._sign_label_en(dominant_sign(curr))} ({curr.get('pos', 0)} pos / "
            f"{curr.get('neg', 0)} neg)"
        )

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
