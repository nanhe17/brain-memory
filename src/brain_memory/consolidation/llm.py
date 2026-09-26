"""LLM 驱动的概念组模式提案。

每个组一次 chat 调用：LLM 看到概念和（截断的）episode，必须返回陈述
*以及真正支持它的 episode 索引*。引擎在确定性代码里对支持子集执行
阈值门——从单条 episode 过度概括的 LLM 直接过不了质量门。传输/解析
失败抛错，引擎降级到确定性巩固器。
"""

from __future__ import annotations

import json

import httpx

from brain_memory.models import Episode, PatternProposal, SemanticKind

_SYSTEM_PROMPT = """You distill a stable piece of knowledge from a set of memory episodes that all mention the same concept.
Return ONLY a JSON object:
{
  "statement": "one self-contained sentence stating the stable pattern/fact/preference (keep the episodes' dominant language)",
  "kind": "fact | preference | schema | generalization",
  "confidence": 0.0-1.0,
  "supporting": [indexes of the episodes that genuinely support this statement]
}
Rules: never invent details; the statement must be backed by at least the minimum support count of episodes listed in "supporting"; if the episodes do not actually share a stable pattern, return {"statement": "", "supporting": []}."""

_RECONSOLIDATION_PROMPT = """You re-examine an existing belief about a concept in light of NEW memory episodes.
Return ONLY a JSON object:
{
  "change_kind": "consistent | contradiction | evolution | correction | context_change",
  "statement": "the updated belief in one self-contained sentence (keep the episodes' dominant language)",
  "kind": "fact | preference | schema | generalization",
  "confidence": 0.0-1.0,
  "supporting": [indexes of the episodes that support the NEW statement]
}
Rules:
- "consistent": the new episodes merely refine or extend the previous statement — write the refined statement.
- any other change_kind: the new evidence changes what we believed. Write a TEMPORAL NARRATIVE statement that preserves the history, e.g. "User used to ... but has now ...". Never erase the past.
- contradiction = the new evidence directly opposes the old belief; evolution = the preference/situation drifted over time; correction = the old statement was simply wrong; context_change = both true in different contexts.
- never invent details; if the new evidence is too thin to update the belief, return {"statement": "", "supporting": []}."""


class LLMConsolidator:
    """LLM 巩固器：新概念提案 + 既有信念的再巩固裁决。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        content_chars: int = 150,
        max_episodes: int = 20,
    ) -> None:
        if not api_key:
            raise ValueError("LLMConsolidator requires an API key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.content_chars = content_chars
        self.max_episodes = max_episodes

    def propose(
        self,
        concept: str,
        episodes: list[Episode],
        *,
        min_support: int,
    ) -> PatternProposal | None:
        """新概念提案：传输/解析失败抛错；LLM 找不到满足阈值的模式返回 None。"""
        window = episodes[: self.max_episodes]
        lines = [
            f"[{index}] ({episode.created_at:%Y-%m-%d}) {episode.content[: self.content_chars]}"
            for index, episode in enumerate(window)
        ]
        user_content = (
            f"CONCEPT: {concept}\nMINIMUM SUPPORT: {min_support}\n\nEPISODES:\n"
            + "\n".join(lines)
        )
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        data = _extract_json(raw)

        statement = str(data.get("statement", "")).strip()
        supporting = _valid_indexes(data.get("supporting"), len(window))
        # 空陈述或支持不足：没有可成型的知识
        if not statement or len(supporting) < min_support:
            return None

        try:
            kind = SemanticKind(str(data.get("kind", "fact")).strip().lower())
        except ValueError:
            kind = SemanticKind.FACT
        confidence = float(data.get("confidence", 0.6))
        return PatternProposal(
            concept=concept,
            statement=" ".join(statement.split()),
            kind=kind if kind is not SemanticKind.CO_OCCURRENCE else SemanticKind.GENERALIZATION,
            confidence=max(0.0, min(1.0, confidence)),
            supporting_indexes=supporting,
        )

    def repropose(
        self,
        concept: str,
        episodes: list[Episode],
        existing,
        *,
        min_support: int,
    ) -> PatternProposal | None:
        """既有语义记忆的再巩固裁决。

        传输/解析失败抛错；证据太薄过不了阈值返回 None（旧陈述保持
        不动）。
        """
        window = episodes[: self.max_episodes]
        lines = [
            f"[{index}] ({episode.created_at:%Y-%m-%d}) {episode.content[: self.content_chars]}"
            for index, episode in enumerate(window)
        ]
        user_content = (
            f"CONCEPT: {concept}\nMINIMUM SUPPORT: {min_support}\n"
            f"PREVIOUS STATEMENT (v{existing.version}): {existing.statement}\n\n"
            "EPISODES:\n" + "\n".join(lines)
        )
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _RECONSOLIDATION_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        data = _extract_json(raw)

        statement = str(data.get("statement", "")).strip()
        supporting = _valid_indexes(data.get("supporting"), len(window))
        if not statement or len(supporting) < min_support:
            return None

        change_kind = str(data.get("change_kind", "consistent")).strip().lower()
        if change_kind not in (
            "consistent", "contradiction", "evolution", "correction", "context_change"
        ):
            change_kind = "consistent"
        try:
            kind = SemanticKind(str(data.get("kind", "fact")).strip().lower())
        except ValueError:
            kind = SemanticKind.FACT
        if kind is SemanticKind.CO_OCCURRENCE:
            kind = SemanticKind.GENERALIZATION
        confidence = float(data.get("confidence", 0.6))
        return PatternProposal(
            concept=concept,
            statement=" ".join(statement.split()),
            kind=kind,
            confidence=max(0.0, min(1.0, confidence)),
            supporting_indexes=supporting,
            change_kind=change_kind,  # type: ignore[arg-type]
        )


def _valid_indexes(value, limit: int) -> list[int]:
    """校验 LLM 返回的支持索引：整数、在范围内、去重。"""
    if not isinstance(value, list):
        return []
    indexes = []
    for item in value:
        if isinstance(item, int) and 0 <= item < limit:
            indexes.append(item)
    return sorted(set(indexes))


def _extract_json(raw: str) -> dict:
    """稳健 JSON 提取：剥代码围栏、截取最外层大括号。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in consolidation response")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("consolidation response is not a JSON object")
    return data
