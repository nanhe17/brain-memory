"""巩固编排器：组 → 提案 → 语义 upsert。

按设计增量推进：巩固状态记住每个概念上次巩固时的 episode 数，因此
只有新证据的组才会被重处理。更新*既有*语义记忆一律走
classify-then-act（repropose）：提案携带 change_kind，非 "consistent"
的裁决产出时间叙事陈述 + 已解决的冲突记录；open 的编码期挑战无论
哪种裁决都会被关闭。每次运行都是显式调用（``engine.consolidate()``）
——调度属于调用方（agent 循环、cron），不在引擎内起后台线程。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from brain_memory.config import MemoryConfig
from brain_memory.consolidation.conflicts import ConflictStore
from brain_memory.consolidation.grouping import CandidateGroup, candidate_groups
from brain_memory.consolidation.heuristic import (
    HeuristicConsolidator,
    polarity_counts,
)
from brain_memory.consolidation.llm import LLMConsolidator
from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.episodic.store import EpisodicStore
from brain_memory.models import (
    ConflictKind,
    ConsolidationReport,
    MemoryConflict,
    PatternProposal,
)

logger = logging.getLogger(__name__)


class Consolidator:
    """巩固编排器。"""

    def __init__(
        self,
        db,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        conflict_store: ConflictStore,
        config: MemoryConfig,
        embed_fn,
    ) -> None:
        self._db = db
        self._episodic = episodic_store
        self._semantic = semantic_store
        self._conflict_store = conflict_store
        self._config = config
        self._embed_fn = embed_fn
        self._heuristic = HeuristicConsolidator()
        self._llm: LLMConsolidator | None = None

    def set_llm(self, llm: LLMConsolidator | None) -> None:
        """引擎在配置了 LLM 时注入 LLM 巩固器。"""
        self._llm = llm

    def consolidate(
        self,
        *,
        max_groups: int = 5,
        max_episodes_per_group: int = 20,
        min_support: int | None = None,
        forced_values: tuple[str, ...] | list[str] = (),
    ) -> ConsolidationReport:
        """运行一轮巩固。

        先收集 open 冲突对应的概念作为强制组，再做候选分组：eligible
        的组逐个提案；提案过质量门则 upsert 语义记忆并推进游标。
        """
        min_support = min_support or self._config.consolidation_min_support
        reconsolidation_min_support = self._config.reconsolidation_min_support
        report = ConsolidationReport()
        now_iso = datetime.now(timezone.utc).isoformat()

        # open 冲突对应的概念强制到期
        forced = set(forced_values)
        for semantic_id in self._conflict_store.open_semantic_ids():
            memory = self._semantic.get(semantic_id)
            if memory is not None:
                forced.add(memory.concept)

        groups = candidate_groups(
            self._db,
            min_support=min_support,
            max_groups=max_groups,
            include_below_support=True,
            forced_values=sorted(forced),
        )
        report.groups_considered = sum(1 for g in groups if g.eligible)

        for group in groups:
            if not group.eligible:
                # 低于阈值也要可见：静默空跑无法调试
                report.skipped.append(
                    f"{group.key}: below min_support ({group.episode_count}/{min_support})"
                )
                continue
            proposal, memory, conflicts = self._consolidate_group(
                group,
                max_episodes_per_group=max_episodes_per_group,
                min_support=min_support,
                reconsolidation_min_support=reconsolidation_min_support,
                now_iso=now_iso,
            )
            if proposal is None or memory is None:
                report.skipped.append(
                    f"{group.key}: no proposal meeting support {min_support}"
                )
                continue
            # 推进游标（按折叠 value 为键）
            self._db.upsert_consolidation_state(
                value=group.value,
                representative_kind=group.kind,
                last_consolidated_at=now_iso,
                episode_count=group.episode_count,
                semantic_id=memory.id,
            )
            (report.created if memory.version == 1 else report.updated).append(memory)
            report.conflicts.extend(conflicts)
        return report

    # -- 单个组 ----------------------------------------------------------------

    def _consolidate_group(
        self,
        group: CandidateGroup,
        *,
        max_episodes_per_group: int,
        min_support: int,
        reconsolidation_min_support: int,
        now_iso: str,
    ):
        """处理一个概念组：无既有记忆走新建，有则走再巩固分类。"""
        episode_ids = self._db.episode_ids_by_tag(group.kind, group.value)
        episodes = [e for e in self._episodic.get_many(episode_ids) if e.is_active]
        episodes.sort(key=lambda e: e.created_at, reverse=True)
        episodes = episodes[:max_episodes_per_group]

        existing = self._semantic.get_by_concept(group.value)
        if existing is None:
            proposal = self._propose(group.value, episodes, min_support)
            if proposal is None:
                return None, None, []
            if not proposal.metadata:
                # 创建时记录极性基线，之后的再巩固才有对比对象
                supporting = [episodes[i] for i in proposal.supporting_indexes]
                proposal = proposal.model_copy(
                    update={"metadata": {"polarity": polarity_counts(supporting)}}
                )
            memory = self._semantic.create(
                concept=group.value,
                kind=proposal.kind,
                statement=proposal.statement,
                confidence=proposal.confidence,
                evidence_ids=sorted({episodes[i].id for i in proposal.supporting_indexes}),
                embedding=self._embed_fn(proposal.statement),
                metadata=dict(proposal.metadata),
                change_reason="initial consolidation",
            )
            return proposal, memory, []

        # 再巩固：对既有信念 classify-then-act
        proposal = self._repropose(
            group.value, episodes, existing, reconsolidation_min_support
        )
        if proposal is None:
            return None, None, []

        evidence = sorted(
            set(existing.evidence_ids) | {episodes[i].id for i in proposal.supporting_indexes}
        )
        smoothed = round(0.5 * existing.confidence + 0.5 * proposal.confidence, 3)
        if proposal.change_kind == "consistent":
            change_reason = f"new evidence ({len(proposal.supporting_indexes)} supporting episodes)"
        else:
            change_reason = f"reconsolidation ({proposal.change_kind})"
        memory = self._semantic.update(
            existing,
            kind=proposal.kind,
            statement=proposal.statement,
            confidence=smoothed,
            evidence_ids=evidence,
            embedding=self._embed_fn(proposal.statement),
            change_reason=change_reason,
            metadata=dict(proposal.metadata),
        )

        conflicts: list[MemoryConflict] = []
        if proposal.change_kind != "consistent":
            # 非 consistent：写冲突记录并立即结清（指向新版本）
            conflict = self._conflict_store.create(
                semantic_id=existing.id,
                kind=ConflictKind(proposal.change_kind),
                old_version=existing.version,
                statement_before=existing.statement,
                trigger_episode_id=None,
                trigger_kind="consolidation",
                metadata={"resolved_inline": True},
            )
            self._conflict_store.resolve_for_semantic(
                existing.id,
                kind=ConflictKind(proposal.change_kind),
                resolution_version=memory.version,
            )
            conflicts.append(self._conflict_store.get(conflict.id))
        else:
            # consistent：未坐实的编码挑战被驳回（保留原 kind 仅关闭）
            self._conflict_store.resolve_for_semantic(
                existing.id,
                kind=ConflictKind.UNRESOLVED,
                resolution_version=memory.version,
                dismiss=True,
            )
        return proposal, memory, [c for c in conflicts if c]

    def _propose(
        self, concept: str, episodes, min_support: int
    ) -> PatternProposal | None:
        """新概念提案：LLM 优先，失败降级启发式。"""
        if self._llm is not None:
            try:
                return self._llm.propose(concept, episodes, min_support=min_support)
            except Exception as exc:  # noqa: BLE001 — 降级到确定性路径
                logger.warning(
                    "LLM consolidation failed for '%s' (%s); using heuristic proposal",
                    concept,
                    exc,
                )
        return self._heuristic.propose(concept, episodes, min_support=min_support)

    def _repropose(
        self, concept: str, episodes, existing, reconsolidation_min_support: int
    ) -> PatternProposal | None:
        """再巩固提案：LLM 优先，**支持门在编排器里确定性执行**
        （原则 5：门绝不能放在可插拔的 LLM 适配器内部）。"""
        if self._llm is not None:
            try:
                proposal = self._llm.repropose(
                    concept, episodes, existing,
                    min_support=reconsolidation_min_support,
                )
            except Exception as exc:  # noqa: BLE001 — 降级到确定性路径
                logger.warning(
                    "LLM reconsolidation failed for '%s' (%s); using heuristic proposal",
                    concept,
                    exc,
                )
            else:
                if proposal is None:
                    return None
                # 支持门：LLM 提案的支持子集不足即拒绝，旧陈述保持不动
                if len(proposal.supporting_indexes) < reconsolidation_min_support:
                    logger.info(
                        "reconsolidation proposal for '%s' failed the support gate "
                        "(%d/%d); keeping the existing statement",
                        concept,
                        len(proposal.supporting_indexes),
                        reconsolidation_min_support,
                    )
                    return None
                return proposal
        return self._heuristic.repropose(
            concept, episodes, existing, min_support=reconsolidation_min_support
        )
