"""确定性提案：诚实的共现陈述，仅此而已。

没有 LLM 时系统无法判断*语义*，所以只陈述它数得出来的东西：某概念
在 N 条记忆中反复出现、这些记忆涉及哪些话题，外加——再巩固时——
偏好标记词指向的*方向*（正面/负面/混合，来自解析器自己的标记词表）。
置信度硬封顶 0.55——刻意低于 LLM 路径能产出的任何值——且 kind 永远
是 ``co_occurrence``。偏好/事实级的知识必须由 LLM 巩固器产出。
"""

from __future__ import annotations

import re
from collections import Counter

from brain_memory.models import Episode, PatternProposal, SemanticKind, SemanticMemory

# 偏好极性标记词（Phase 4 再巩固复用）
_POSITIVE_MARKERS = ("喜欢", "偏好", "偏爱", "热衷", "love", "like", "prefer", "enjoy")
_NEGATIVE_MARKERS = ("不喜欢", "讨厌", "反感", "不想", "hate", "dislike", "not a fan")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _cjk_ratio(text: str) -> float:
    """中文字符占比（选择陈述模板语言用）。"""
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / max(len(text), 1)


def episode_polarity(content: str) -> str | None:
    """单条 episode 的偏好方向：'pos' | 'neg' | 'mixed' | None。

    负面标记先检出并清洗，再查正面标记——避免「不喜欢」被误计成
    含有「喜欢」。
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
    """统计一组 episode 的极性分布 {pos, neg, mixed}。"""
    counts = {"pos": 0, "neg": 0, "mixed": 0}
    for episode in episodes:
        sign = episode_polarity(episode.content)
        if sign:
            counts[sign] += 1
    return counts


def dominant_sign(counts: dict[str, int]) -> str | None:
    """极性分布的主导符号：'pos' | 'neg' | 'mixed' | None（全零）。"""
    if counts["pos"] > counts["neg"]:
        return "pos"
    if counts["neg"] > counts["pos"]:
        return "neg"
    if counts["pos"] + counts["neg"] > 0:
        return "mixed"
    return None


class HeuristicConsolidator:
    """仅做统计提案；策略见模块 docstring。"""

    def propose(
        self,
        concept: str,
        episodes: list[Episode],
        *,
        min_support: int,
    ) -> PatternProposal | None:
        """共现提案：支持数不足 min_support 直接返回 None。"""
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

        # 依内容语言选择中/英陈述模板
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
        """既有记忆的再巩固提案：同样的诚实计数 + 偏好极性跟踪。

        跨版本的主导符号变化（正/负/混合三态）产出演化陈述；首次观察到
        极性没有基线，永远视为 'consistent'。
        """
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
        """概念是否出现在该 episode 的标签或正文中。"""
        tags = {t.casefold() for t in [*episode.entities, *episode.topics]}
        return concept.casefold() in tags or concept.casefold() in episode.content.casefold()

    @staticmethod
    def _top_topics(concept: str, episodes: list[Episode], limit: int = 4) -> list[str]:
        """按频次统计支撑集的话题（排除概念自身），取前 limit 个。"""
        counter: Counter[str] = Counter()
        for episode in episodes:
            for tag in {t.casefold() for t in [*episode.entities, *episode.topics]}:
                if tag != concept.casefold():
                    counter[tag] += 1
        # 频率优先，再按名称确定序
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        return [topic for topic, _count in ranked[:limit]]

    @staticmethod
    def _sign_label_zh(sign: str) -> str:
        """极性符号的中文标签。"""
        return {"pos": "正面", "neg": "负面", "mixed": "正负并存"}[sign]

    @staticmethod
    def _sign_label_en(sign: str) -> str:
        """极性符号的英文标签。"""
        return {"pos": "positive", "neg": "negative", "mixed": "mixed"}[sign]

    @classmethod
    def _shift_statement_zh(cls, concept: str, prev: dict, curr: dict) -> str:
        """极性反转的中文叙事陈述（含前后计数）。"""
        return (
            f"{concept} 的态度信号出现变化：此前以{cls._sign_label_zh(dominant_sign(prev))}为主"
            f"（{prev.get('pos', 0)} 正 / {prev.get('neg', 0)} 负），"
            f"近期以{cls._sign_label_zh(dominant_sign(curr))}为主"
            f"（{curr.get('pos', 0)} 正 / {curr.get('neg', 0)} 负）"
        )

    @classmethod
    def _shift_statement_en(cls, concept: str, prev: dict, curr: dict) -> str:
        """极性反转的英文叙事陈述。"""
        return (
            f"{concept} attitude signals shifted: earlier mostly "
            f"{cls._sign_label_en(dominant_sign(prev))} ({prev.get('pos', 0)} pos / "
            f"{prev.get('neg', 0)} neg), recently mostly "
            f"{cls._sign_label_en(dominant_sign(curr))} ({curr.get('pos', 0)} pos / "
            f"{curr.get('neg', 0)} neg)"
        )

    @staticmethod
    def _statement_zh(concept: str, count: int, topics: list[str]) -> str:
        """中文共现陈述模板。"""
        head = f"{concept} 在 {count} 条记忆中反复出现"
        if topics:
            return f"{head}，主要涉及 {'、'.join(topics)}"
        return head

    @staticmethod
    def _statement_en(concept: str, count: int, topics: list[str]) -> str:
        """英文共现陈述模板。"""
        head = f"{concept} shows up across {count} memories"
        if topics:
            return f"{head}, mainly involving {', '.join(topics)}"
        return head
