"""Deterministic, zero-dependency experience parser.

This is the default parser and the system's ground truth: it never calls a
network, never fails on odd input, and always produces the same output for
the same input.  It is deliberately conservative:

* entities — quoted spans, known tech lexicon (latin + CJK), CamelCase /
  alphanumeric tokens; noisy generic words are stoplisted;
* topics — lexicon-canonical tags;
* key facts — sentences carrying fact markers;
* importance / confidence — additive heuristics over emphasis signals.

An LLM parser can refine all of this later (see ``llm.py``), but the system
must stay useful with heuristics alone (design doc, Principle 5).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from brain_memory.extraction.base import ExtractedExperience

# --- lexicons ---------------------------------------------------------------

_LATIN_LEXICON = {
    "python", "java", "javascript", "typescript", "rust", "golang", "c++", "c#",
    "spring", "spring boot", "mysql", "postgres", "postgresql", "sqlite", "redis",
    "mongodb", "docker", "kubernetes", "fastapi", "django", "flask", "pytorch",
    "tensorflow", "llm", "llms", "agent", "agents", "memory", "embedding",
    "embeddings", "vector", "rag", "api", "http", "rest", "grpc", "github",
    "git", "linux", "windows", "minecraft", "forge", "mod", "shader", "unity",
    "unreal", "blender", "zcode", "openai", "glm", "zhipu", "p51", "mustang",
}

_CJK_LEXICON = [
    "飞机", "模组", "游戏", "服务器", "数据库", "向量", "嵌入", "智能体", "代理",
    "记忆", "情景记忆", "语义记忆", "工作记忆", "巩固", "遗忘", "检索", "项目",
    "学习", "后端", "前端", "微服务", "架构", "开源", "渲染", "建模", "测试",
    "部署", "爬虫", "算法", "模型", "训练", "推理", "提示词", "上下文",
]

_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")
_QUOTED_RE = re.compile(r"[\"“”「『']([^\"“”」』']{1,40})[\"“”」』']")

_FACT_MARKERS = [
    "是", "有", "喜欢", "讨厌", "想要", "打算", "决定", "完成", "需要", "觉得",
    "认为", "在学", "正在", "已经", "计划", "用", "做了", "要", "不想", "偏好",
    "want", "need", "like", "prefer", "decide", "use", "built", "finish",
    "learning", "working on", "plan",
]

_PREFERENCE_MARKERS = ["喜欢", "讨厌", "偏好", "不想", "love", "hate", "like", "prefer", "dislike"]

_SIGNAL_PATTERNS = [
    ("explicit_remember", re.compile(r"记住|记一下|别忘|remember|don't forget|do not forget", re.I)),
    ("importance_marker", re.compile(r"重要|关键|critical|important", re.I)),
    ("correction", re.compile(r"其实|不对|错了|搞错|不再|改成|not\s+\w+\s+anymore|no longer|instead", re.I)),
    ("question", re.compile(r"[?？]\s*$")),
]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "to", "of", "in", "on", "for", "with", "at", "by",
    "from", "it", "its", "this", "that", "these", "those", "i", "im", "ive",
    "you", "your", "we", "our", "they", "he", "she", "my", "me", "do", "does",
    "did", "not", "no", "yes", "so", "as", "can", "will", "would", "should",
    "have", "has", "had", "get", "got", "about", "into", "over", "out", "up",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class HeuristicExperienceParser:
    """Deterministic parser; see module docstring for the strategy."""

    def parse(
        self,
        text: str,
        *,
        source: str = "conversation",
        context: str | None = None,
        timestamp: datetime | None = None,
    ) -> ExtractedExperience:
        content = " ".join((text or "").split())
        if not content:
            raise ValueError("cannot parse empty experience text")

        lower = content.lower()
        entities = self._extract_entities(content)
        topics = self._extract_topics(lower)
        key_facts = self._extract_key_facts(content, entities)
        signals = self._detect_signals(content, entities)

        importance = self._score_importance(signals, entities, lower)
        confidence = self._score_confidence(signals, entities, key_facts)

        return ExtractedExperience(
            content=content,
            entities=entities,
            topics=topics,
            key_facts=key_facts,
            emphasis_signals=[name for name, _ in signals],
            context=context,
            source=source,
            importance=importance,
            confidence=confidence,
            timestamp=timestamp or datetime.now(timezone.utc),
        )

    # -- entities ------------------------------------------------------------

    def _extract_entities(self, content: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            value = value.strip().strip(".,;:!?，。；：！？")
            key = value.casefold()
            if len(value) < 2 or key in seen or key in _STOPWORDS:
                return
            seen.add(key)
            found.append(value)

        for match in _QUOTED_RE.finditer(content):
            add(match.group(1))

        for term in _CJK_LEXICON:
            if term in content:
                add(term)

        for match in _LATIN_TOKEN_RE.finditer(content):
            token = match.group(0)
            key = token.lower().rstrip(".-")
            if key in _LATIN_LEXICON or key.rstrip("+#") in _LATIN_LEXICON:
                add(token.rstrip(".-"))
                continue
            has_inner_upper = any(c.isupper() for c in token[1:])
            has_digit = any(c.isdigit() for c in token)
            if (has_inner_upper or has_digit) and key not in _STOPWORDS:
                add(token.rstrip(".-"))

        return found

    def _extract_topics(self, lower_content: str) -> list[str]:
        topics: list[str] = []
        for term in sorted(_LATIN_LEXICON):
            if re.search(rf"\b{re.escape(term)}\b", lower_content):
                topics.append(term)
        for term in _CJK_LEXICON:
            if term in lower_content:
                topics.append(term)
        return topics

    # -- facts & signals -------------------------------------------------------

    def _extract_key_facts(self, content: str, entities: list[str]) -> list[str]:
        sentences = [s.strip() for s in re.split(r"[.!?。！？;\n;]+", content) if s.strip()]
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            score = 0
            low = sentence.lower()
            if any(marker in low or marker in sentence for marker in _FACT_MARKERS):
                score += 2
            score += sum(1 for entity in entities if entity in sentence)
            if score > 0:
                scored.append((score, sentence))
        scored.sort(key=lambda pair: -pair[0])
        return [sentence for _, sentence in scored[:3]]

    def _detect_signals(self, content: str, entities: list[str]) -> list[tuple[str, float]]:
        signals: list[tuple[str, float]] = []
        for name, pattern in _SIGNAL_PATTERNS:
            if pattern.search(content):
                signals.append((name, 1.0))
        if any(ch in content for ch in ("!", "！")):
            signals.append(("exclamation", 1.0))
        repeated = [
            entity
            for entity in entities
            if len(re.findall(re.escape(entity), content, re.I)) >= 2
        ]
        for entity in repeated[:2]:
            signals.append((f"repetition:{entity}", 1.0))
        return signals

    # -- scoring -----------------------------------------------------------------

    def _score_importance(self, signals: list[tuple[str, float]], entities: list[str],
                          lower: str) -> float:
        names = {name for name, _ in signals}
        importance = 0.30
        if "explicit_remember" in names:
            importance += 0.35
        if "importance_marker" in names:
            importance += 0.20
        if "correction" in names:
            importance += 0.15
        if "exclamation" in names:
            importance += 0.05
        if any(marker in lower for marker in _PREFERENCE_MARKERS):
            importance += 0.10
        importance += min(0.10, 0.02 * len(entities))
        if "question" in names and "explicit_remember" not in names:
            importance -= 0.10
        return round(_clamp(importance, 0.05, 1.0), 3)

    def _score_confidence(self, signals: list[tuple[str, float]], entities: list[str],
                          key_facts: list[str]) -> float:
        names = {name for name, _ in signals}
        confidence = 0.50
        if "explicit_remember" in names:
            confidence += 0.20
        confidence += 0.10 if entities else 0.0
        confidence += 0.10 if key_facts else 0.0
        return round(_clamp(confidence, 0.30, 0.95), 3)
