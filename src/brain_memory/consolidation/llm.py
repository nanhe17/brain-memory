"""LLM-backed pattern proposal for one concept group.

One chat call per group: the LLM sees the concept and the (truncated)
episodes and must return a statement *plus the indexes of the episodes that
genuinely support it*.  The engine enforces the support threshold on that
subset — an LLM that over-generalizes from one episode simply fails the
quality gate.  Transport/parse failures raise; the engine falls back to the
deterministic consolidator.
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


class LLMConsolidator:
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
        """Raises on transport/parse failure; returns None when the LLM finds
        no pattern that meets the support threshold."""
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


def _valid_indexes(value, limit: int) -> list[int]:
    if not isinstance(value, list):
        return []
    indexes = []
    for item in value:
        if isinstance(item, int) and 0 <= item < limit:
            indexes.append(item)
    return sorted(set(indexes))


def _extract_json(raw: str) -> dict:
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
