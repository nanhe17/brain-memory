"""Optional LLM-backed experience parser (OpenAI-compatible /chat/completions).

Used only when explicitly configured (``MEMORY_LLM_MODEL`` + key).  Every
failure mode — network errors, malformed JSON, schema violations — degrades
to the heuristic parser, because encoding must never lose an experience
(design doc, Principle 5: LLM assists, deterministic code owns the pipeline).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from brain_memory.extraction.base import ExtractedExperience, ExperienceParser

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You extract structured memories for an AI agent's memory system.
Return ONLY a JSON object with these fields:
{
  "content": "cleaned, self-contained summary of what happened (keep the user's language)",
  "entities": ["concrete nouns: technologies, projects, people, products"],
  "topics": ["1-3 lowercase topical tags"],
  "key_facts": ["short factual statements asserted in the text, max 3"],
  "emphasis_signals": ["explicit_remember | importance_marker | correction | exclamation | repetition"],
  "importance": 0.0-1.0,
  "confidence": 0.0-1.0
}
Rules: never invent facts; importance higher for explicit requests to remember, corrections, strong preferences; importance lower for small talk and questions."""


class LLMExperienceParser:
    """Heuristic parser + LLM refinement, with graceful degradation."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        fallback: ExperienceParser,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("LLMExperienceParser requires an API key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.fallback = fallback
        self.timeout = timeout

    def parse(
        self,
        text: str,
        *,
        source: str = "conversation",
        context: str | None = None,
        timestamp: datetime | None = None,
    ) -> ExtractedExperience:
        try:
            extracted = self._parse_with_llm(text)
        except Exception as exc:  # noqa: BLE001 — degrade, never lose the experience
            logger.warning("LLM parse failed (%s); falling back to heuristic parser", exc)
            return self.fallback.parse(text, source=source, context=context, timestamp=timestamp)
        extracted.source = source
        extracted.context = context
        extracted.timestamp = timestamp or datetime.now(timezone.utc)
        return extracted

    def _parse_with_llm(self, text: str) -> ExtractedExperience:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        data = self._extract_json(raw)
        return self._validate(data, text)

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in LLM response")
        data = json.loads(raw[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")
        return data

    @staticmethod
    def _validate(data: dict[str, Any], original: str) -> ExtractedExperience:
        content = str(data.get("content") or "").strip() or original.strip()
        if not content:
            raise ValueError("LLM returned empty content")
        importance = float(data.get("importance", 0.3))
        confidence = float(data.get("confidence", 0.6))
        return ExtractedExperience(
            content=" ".join(content.split()),
            entities=[str(e) for e in data.get("entities", [])][:20],
            topics=[str(t).lower() for t in data.get("topics", [])][:10],
            key_facts=[str(f) for f in data.get("key_facts", [])][:5],
            emphasis_signals=[str(s) for s in data.get("emphasis_signals", [])][:10],
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
        )
