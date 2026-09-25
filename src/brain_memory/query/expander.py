"""LLM-backed query expansion for recall cues.

An expansion maps a possibly vague cue ("那个模组怎么样了") into retrieval-ready
variants, using the working-memory snapshot to resolve references.  Design
rules:

* The expansion is *advisory*: the engine always retrieves for the original
  cue too, so a bad rewrite can only add candidates, never remove them.
* Every failure mode (network, malformed JSON, schema violation) returns
  ``None`` — recall must degrade to the Phase-1 behavior, never fail.
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
    rewritten: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    time_from: str | None = None
    time_to: str | None = None

    @field_validator("rewritten", "sub_queries", mode="before")
    @classmethod
    def _strip_strings(cls, value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value


class LLMQueryExpander:
    """One chat call per expand; degrades to ``None`` on any failure."""

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
        try:
            return self._expand(cue, working_context)
        except Exception as exc:  # noqa: BLE001 — expansion is advisory
            logger.warning("query expansion failed (%s); using original cue", exc)
            return None

    def _expand(self, cue: str, working_context: str) -> QueryExpansion | None:
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
        if not expansion.rewritten:
            return None
        return expansion


def _extract_json(raw: str) -> dict[str, Any]:
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
