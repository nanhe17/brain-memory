"""确定性、零依赖的经验解析器。

这是默认解析器，也是系统的事实基准：不访问网络、不在异常输入上失败、
相同输入永远产出相同输出。刻意保持保守：

* 实体——引号片段、已知技术词典（拉丁 + 中文）、驼峰/含数字词元；
  噪声大的通用词进停用词表；
* 话题——词典规范化的标签；
* 关键事实——携带事实标记词的句子；
* 重要性/置信度——对强调信号做加法启发式。

LLM 解析器之后可以全面精炼（见 ``llm.py``），但系统必须只用启发式就
保持可用（设计文档原则 5）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from brain_memory.extraction.base import ExtractedExperience

# --- 词典 -------------------------------------------------------------------

# 拉丁技术词（小写匹配）
_LATIN_LEXICON = {
    "python", "java", "javascript", "typescript", "rust", "golang", "c++", "c#",
    "spring", "spring boot", "mysql", "postgres", "postgresql", "sqlite", "redis",
    "mongodb", "docker", "kubernetes", "fastapi", "django", "flask", "pytorch",
    "tensorflow", "llm", "llms", "agent", "agents", "memory", "embedding",
    "embeddings", "vector", "rag", "api", "http", "rest", "grpc", "github",
    "git", "linux", "windows", "minecraft", "forge", "mod", "shader", "unity",
    "unreal", "blender", "zcode", "openai", "glm", "zhipu", "p51", "mustang",
}

# 中文概念词（子串匹配）
_CJK_LEXICON = [
    "飞机", "模组", "游戏", "服务器", "数据库", "向量", "嵌入", "智能体", "代理",
    "记忆", "情景记忆", "语义记忆", "工作记忆", "巩固", "遗忘", "检索", "项目",
    "学习", "后端", "前端", "微服务", "架构", "开源", "渲染", "建模", "测试",
    "部署", "爬虫", "算法", "模型", "训练", "推理", "提示词", "上下文",
]

_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")
_QUOTED_RE = re.compile(r"[\"“”「『']([^\"“”」』']{1,40})[\"“”」』']")

# 事实标记词：出现在句中即视为"事实陈述"
_FACT_MARKERS = [
    "是", "有", "喜欢", "讨厌", "想要", "打算", "决定", "完成", "需要", "觉得",
    "认为", "在学", "正在", "已经", "计划", "用", "做了", "要", "不想", "偏好",
    "want", "need", "like", "prefer", "decide", "use", "built", "finish",
    "learning", "working on", "plan",
]

# 偏好标记词（Phase 4 极性统计复用）
_PREFERENCE_MARKERS = ["喜欢", "讨厌", "偏好", "不想", "love", "hate", "like", "prefer", "dislike"]

# 强调信号模式：显式记忆请求 / 重要标记 / 纠正 / 疑问
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
    """数值截断到 [low, high]。"""
    return max(low, min(high, value))


class HeuristicExperienceParser:
    """确定性解析器；策略见模块 docstring。"""

    def parse(
        self,
        text: str,
        *,
        source: str = "conversation",
        context: str | None = None,
        timestamp: datetime | None = None,
    ) -> ExtractedExperience:
        """解析主流程：清洗 → 实体/话题/事实/信号 → 重要性/置信度打分。"""
        # 规范化空白，空文本直接拒绝（调用方决定如何降级）
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

    # -- 实体提取 ------------------------------------------------------------

    def _extract_entities(self, content: str) -> list[str]:
        """实体提取：引号片段 → 中文词典 → 拉丁词元（词典/驼峰/含数字）。"""
        found: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            # 清理边缘标点；大小写折叠去重；停用词与过短词排除
            value = value.strip().strip(".,;:!?，。；：！？")
            key = value.casefold()
            if len(value) < 2 or key in seen or key in _STOPWORDS:
                return
            seen.add(key)
            found.append(value)

        # 1) 引号片段是最强信号，直接采信
        for match in _QUOTED_RE.finditer(content):
            add(match.group(1))

        # 2) 中文词典命中
        for term in _CJK_LEXICON:
            if term in content:
                add(term)

        # 3) 拉丁词元：词典命中 / 驼峰 / 含数字
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
        """话题提取：拉丁词典按词边界匹配 + 中文词典子串匹配。"""
        topics: list[str] = []
        # sorted 保证跨进程顺序确定（set 迭代序受哈希随机化影响）
        for term in sorted(_LATIN_LEXICON):
            if re.search(rf"\b{re.escape(term)}\b", lower_content):
                topics.append(term)
        for term in _CJK_LEXICON:
            if term in lower_content:
                topics.append(term)
        return topics

    # -- 事实与信号 -------------------------------------------------------

    def _extract_key_facts(self, content: str, entities: list[str]) -> list[str]:
        """关键事实：按事实标记词 + 实体命中数给句子打分，取前 3 句。"""
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
        """强调信号检测（纠正/重要/疑问/感叹/实体重复）。"""
        signals: list[tuple[str, float]] = []
        for name, pattern in _SIGNAL_PATTERNS:
            if pattern.search(content):
                signals.append((name, 1.0))
        if any(ch in content for ch in ("!", "！")):
            signals.append(("exclamation", 1.0))
        # 出现 ≥2 次的实体视为"被强调"
        repeated = [
            entity
            for entity in entities
            if len(re.findall(re.escape(entity), content, re.I)) >= 2
        ]
        for entity in repeated[:2]:
            signals.append((f"repetition:{entity}", 1.0))
        return signals

    # -- 打分 -----------------------------------------------------------------

    def _score_importance(self, signals: list[tuple[str, float]], entities: list[str],
                          lower: str) -> float:
        """重要性启发式：基线 0.3，按强调信号加权；纯提问降权。"""
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
        """置信度启发式：结构化信号越充分越可信。"""
        names = {name for name, _ in signals}
        confidence = 0.50
        if "explicit_remember" in names:
            confidence += 0.20
        confidence += 0.10 if entities else 0.0
        confidence += 0.10 if key_facts else 0.0
        return round(_clamp(confidence, 0.30, 0.95), 3)
