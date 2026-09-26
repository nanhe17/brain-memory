"""LLM 驱动的召回 cue 扩展。

扩展把一个可能含糊的 cue（"那个模组怎么样了"）映射为可直接检索的
变体，并借助工作记忆快照消解指代。设计规则：

* 扩展只是*建议*：引擎永远同时对原始 cue 检索，所以糟糕的改写只会
  增加候选，绝不会移除它们；
* 所有失败形态（网络、畸形 JSON、schema 违规）返回 ``None``——召回
  必须降级到 Phase 1 行为，绝不能失败。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You rewrite memory-retrieval cues for an AI agent's memory system.
Return ONLY a JSON object:
{
  "rewritten": "the cue rewritten as a self-contained search query, resolving pronouns and references via the context; keep the user's language",
  "sub_queries": ["0-2 alternative short search queries worth trying"],
  "entities": ["concrete entities the rewritten query is about"],
  "time_from": null,
  "time_to": null
}
Set time_from/time_to (ISO 8601) only when the cue clearly refers to a time range ("yesterday", "last week"); otherwise null.  Never invent entities that are neither in the cue nor in the context."""


class QueryExpansion(BaseModel):
    """一次查询扩展的结果。"""

    rewritten: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    time_from: str | None = None
    time_to: str | None = None

    @field_validator("rewritten", "sub_queries", mode="before")
    @classmethod
    def _strip_strings(cls, value):
        """字符串剥空白；列表项转字符串并剔除空项。"""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value


class LLMQueryExpander:
    """每次 expand 一次 chat 调用；任何失败降级为 ``None``。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 15.0,
    ) -> None:
        if not api_key:
            raise ValueError("LLMQueryExpander requires an API key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def expand(self, cue: str, working_context: str) -> QueryExpansion | None:
        """扩展入口：失败一律返回 None（调用方继续用原始 cue）。"""
        try:
            return self._expand(cue, working_context)
        except Exception as exc:  # noqa: BLE001 — 扩展仅是建议
            logger.warning("query expansion failed (%s); using original cue", exc)
            return None

    def _expand(self, cue: str, working_context: str) -> QueryExpansion | None:
        """实际的 chat 调用与 JSON 解析。"""
        user_content = f"CONTEXT:\n{working_context or '(none)'}\n\nCUE: {cue}"
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
        expansion = QueryExpansion.model_validate(data)
        # 改写为空视同没有扩展
        if not expansion.rewritten:
            return None
        return expansion


def _extract_json(raw: str) -> dict[str, Any]:
    """稳健 JSON 提取：剥代码围栏、截取最外层大括号。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in expansion response")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("expansion response is not a JSON object")
    return data
