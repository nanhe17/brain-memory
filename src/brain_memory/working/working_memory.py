"""工作记忆——会话级的"agent 正在做什么"容器。

它持有当前目标、活跃实体、最近触碰的记忆和待解问题，并在 token 预算
内把自己渲染成 prompt 块。刻意不做持久化：工作记忆是状态，不是数据库
（设计文档 §5——绝不把整个长期记忆库塞进 prompt）。
"""

from __future__ import annotations

import re
from collections import deque

from brain_memory.models import RecallResult, WorkingMemoryState

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """粗略的中英混合 token 估算：中文约 1 token/字，拉丁约 0.3 token/字符。"""
    cjk = len(_CJK_RE.findall(text))
    latin = len(text) - cjk
    latin_tokens = round(latin / 3.5) if latin else 0
    return cjk + max(latin_tokens, 1 if latin else 0)


class WorkingMemory:
    """每个 agent 会话一个实例；创建与丢弃都很廉价。"""

    def __init__(self, token_budget: int = 2000, recent_window: int = 20) -> None:
        self.state = WorkingMemoryState(token_budget=token_budget)
        self._recent: deque[int] = deque(maxlen=recent_window)
        self._last_recall: list[RecallResult] = []

    # -- 状态更新 ---------------------------------------------------------

    def set_goal(self, goal: str) -> None:
        """设置当前目标。"""
        self.state.current_goal = goal

    def add_entities(self, entities: list[str]) -> None:
        """合并活跃实体（去重，封顶 32 个防漂移）。"""
        for entity in entities:
            if entity not in self.state.active_entities:
                self.state.active_entities.append(entity)
        del self.state.active_entities[32:]

    def note_episode(self, episode_id: int) -> None:
        """记录一条 episode 属于当前对话。"""
        if episode_id not in self._recent:
            self._recent.append(episode_id)
        self.state.recent_episode_ids = list(self._recent)

    def remember_recall(self, results: list[RecallResult]) -> None:
        """保存最近一次召回：更新检索结果、最近 episode 与活跃实体。"""
        self._last_recall = results
        self.state.retrieved_memory_ids = [r.episode.id for r in results]
        for result in results:
            self.note_episode(result.episode.id)
            self.add_entities(result.episode.entities)

    def add_question(self, question: str) -> None:
        """记录一个待解问题（去重）。"""
        if question not in self.state.unresolved_questions:
            self.state.unresolved_questions.append(question)

    def resolve_question(self, question: str) -> None:
        """标记一个问题已解决。"""
        self.state.unresolved_questions = [
            q for q in self.state.unresolved_questions if q != question
        ]

    # -- prompt 渲染 ---------------------------------------------------------

    @property
    def last_recall(self) -> list[RecallResult]:
        """最近一次召回的结果副本。"""
        return list(self._last_recall)

    def snapshot(self) -> str:
        """当前状态的紧凑文本（供 LLM 查询扩展作上下文）。"""
        lines: list[str] = []
        if self.state.current_goal:
            lines.append(f"goal: {self.state.current_goal}")
        if self.state.active_entities:
            lines.append(f"active entities: {', '.join(self.state.active_entities[:10])}")
        if self.state.unresolved_questions:
            lines.append(f"open questions: {'; '.join(self.state.unresolved_questions[:3])}")
        return "\n".join(lines)

    def build_prompt_block(self, *, token_budget: int | None = None) -> str:
        """把记忆渲染为可注入 LLM prompt 的块，遵守预算。

        得分更高的召回记忆先输出；预算耗尽时丢弃剩余（绝不截断半条
        记忆——半条记忆比没有更糟）。语义命中渲染在 [KNOWN FACTS] 段，
        情景命中在 [RELEVANT MEMORIES] 段。
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

        # 语义命中：巩固知识段
        semantic_hits = [r for r in self._last_recall if r.is_semantic and r.semantic]
        if semantic_hits:
            lines.append("[KNOWN FACTS]")
            used += estimate_tokens("[KNOWN FACTS]")
            for result in semantic_hits:
                memory = result.semantic
                entry = (
                    f"- {memory.concept}: {memory.statement} "
                    f"(S-{memory.id}, confidence {memory.confidence:.2f})"
                )
                cost = estimate_tokens(entry)
                if used + cost > budget:
                    break
                lines.append(entry)
                used += cost

        # 情景命中：相关记忆段
        episodic_hits = [r for r in self._last_recall if not r.is_semantic]
        if episodic_hits:
            lines.append("[RELEVANT MEMORIES]")
            used += estimate_tokens("[RELEVANT MEMORIES]")
            for result in episodic_hits:
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
