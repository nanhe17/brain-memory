"""Working memory — a session-scoped container for what the agent is doing.

It holds the current goal, active entities, recently touched memories, and
open questions, and renders itself into a prompt block under a token budget.
It deliberately owns no persistence: working memory is state, not a database
(design doc §5 — never paste the whole long-term store into the prompt).
"""

from __future__ import annotations

import re
from collections import deque

from brain_memory.models import RecallResult, WorkingMemoryState

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """Rough mixed-script estimate: ~1 token per CJK char, ~0.3 per latin char."""
    cjk = len(_CJK_RE.findall(text))
    latin = len(text) - cjk
    latin_tokens = round(latin / 3.5) if latin else 0
    return cjk + max(latin_tokens, 1 if latin else 0)


class WorkingMemory:
    """One instance per agent session; cheap to create and to throw away."""

    def __init__(self, token_budget: int = 2000, recent_window: int = 20) -> None:
        self.state = WorkingMemoryState(token_budget=token_budget)
        self._recent: deque[int] = deque(maxlen=recent_window)
        self._last_recall: list[RecallResult] = []

    # -- state updates ---------------------------------------------------------

    def set_goal(self, goal: str) -> None:
        self.state.current_goal = goal

    def add_entities(self, entities: list[str]) -> None:
        for entity in entities:
            if entity not in self.state.active_entities:
                self.state.active_entities.append(entity)
        del self.state.active_entities[32:]

    def note_episode(self, episode_id: int) -> None:
        """Record that an episode is part of the current conversation."""
        if episode_id not in self._recent:
            self._recent.append(episode_id)
        self.state.recent_episode_ids = list(self._recent)

    def remember_recall(self, results: list[RecallResult]) -> None:
        self._last_recall = results
        self.state.retrieved_memory_ids = [r.episode.id for r in results]
        for result in results:
            self.note_episode(result.episode.id)
            self.add_entities(result.episode.entities)

    def add_question(self, question: str) -> None:
        if question not in self.state.unresolved_questions:
            self.state.unresolved_questions.append(question)

    def resolve_question(self, question: str) -> None:
        self.state.unresolved_questions = [
            q for q in self.state.unresolved_questions if q != question
        ]

    # -- prompt rendering ---------------------------------------------------------

    @property
    def last_recall(self) -> list[RecallResult]:
        return list(self._last_recall)

    def build_prompt_block(self, *, token_budget: int | None = None) -> str:
        """Render memories for injection into an LLM prompt, within budget.

        Higher-scored recalled memories are emitted first; when the budget
        runs out, the rest are dropped (never truncated mid-memory — a
        half-memory is worse than an absent one).
        """
        budget = token_budget or self.state.token_budget
        lines: list[str] = []
        used = 0

        if self.state.current_goal:
            line = f"[GOAL] {self.state.current_goal}"
            lines.append(line)
            used += estimate_tokens(line)
        if self.state.active_entities:
            line = f"[ACTIVE] {', '.join(self.state.active_entities[:12])}"
            lines.append(line)
            used += estimate_tokens(line)
        if self.state.unresolved_questions:
            line = f"[OPEN QUESTIONS] {'; '.join(self.state.unresolved_questions[:5])}"
            lines.append(line)
            used += estimate_tokens(line)

        if self._last_recall:
            lines.append("[RELEVANT MEMORIES]")
            used += estimate_tokens("[RELEVANT MEMORIES]")
            for result in self._last_recall:
                episode = result.episode
                entry = (
                    f"- #{episode.id} ({episode.created_at:%Y-%m-%d}) "
                    f"{episode.content}  <why: {'; '.join(result.reasons[:2]) or 'matched'}>"
                )
                cost = estimate_tokens(entry)
                if used + cost > budget:
                    break
                lines.append(entry)
                used += cost

        return "\n".join(lines) if lines else ""
