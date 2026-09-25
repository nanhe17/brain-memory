"""Blind listwise LLM reranking of retrieval candidates.

The reranker sees only the cue and the candidate contents — never the factor
scores — so its judgment stays independent (no anchoring), and the fusion
keeps recency/importance influence that a text-only judge cannot see:

    final = mix * llm_relevance + (1 - mix) * factor_score

Missing candidate ids score a neutral 0.5; transport or parse failures raise,
and the engine falls back to the pure factor ranking.
"""

from __future__ import annotations

import json
from typing import Protocol

import httpx

from brain_memory.models import RecallResult

_SYSTEM_PROMPT = """You judge how topically relevant each candidate memory is to a query.
Score each candidate with a relevance between 0.0 and 1.0:
1.0 = directly about the same thing, 0.5 = related but not the same matter, 0.0 = unrelated.
Judge content only — you know nothing else about the candidates.
Return ONLY a JSON array: [{"id": <candidate number>, "relevance": <0.0-1.0>}, ...]"""


class Reranker(Protocol):
    name: str

    def rerank(self, cue: str, candidates: list[RecallResult]) -> list[float]:
        """Relevance scores aligned with the input order."""
        ...


class LLMReranker:
    """Single listwise call; raises on failure (engine degrades to factors)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 8.0,
        retries: int = 1,
        content_chars: int = 200,
    ) -> None:
        if not api_key:
            raise ValueError("LLMReranker requires an API key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.content_chars = content_chars

    @property
    def name(self) -> str:
        return f"llm_rerank:{self.model}"

    def rerank(self, cue: str, candidates: list[RecallResult]) -> list[float]:
        lines = [
            f"[{index}] ({result.episode.created_at:%Y-%m-%d}) "
            f"{result.episode.content[: self.content_chars]}"
            for index, result in enumerate(candidates)
        ]
        user_content = f"QUERY: {cue}\n\nCANDIDATES:\n" + "\n".join(lines)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
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
                return self._parse_scores(raw, len(candidates))
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(f"rerank failed: {last_error}")

    @staticmethod
    def _parse_scores(raw: str, count: int) -> list[float]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("no JSON array in rerank response")
        items = json.loads(raw[start : end + 1])
        if not isinstance(items, list):
            raise ValueError("rerank response is not a JSON array")

        neutral = 0.5
        scores = [neutral] * count
        for item in items:
            if not isinstance(item, dict):
                continue
            index = item.get("id")
            relevance = item.get("relevance")
            if not isinstance(index, int) or not 0 <= index < count:
                continue
            if not isinstance(relevance, (int, float)):
                continue
            scores[index] = max(0.0, min(1.0, float(relevance)))
        return scores
