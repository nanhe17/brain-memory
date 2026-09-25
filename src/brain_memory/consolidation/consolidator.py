"""The consolidation orchestrator: groups -> proposals -> semantic upserts.

Incremental by design: ``consolidation_state`` remembers how many episodes
each group had when last consolidated, so only groups with new evidence are
reprocessed.  Every run is explicit (``engine.consolidate()``) — scheduling
belongs to the caller (agent loop, cron), not to a background thread inside
the engine.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from brain_memory.config import MemoryConfig
from brain_memory.consolidation.grouping import CandidateGroup, candidate_groups
from brain_memory.consolidation.heuristic import HeuristicConsolidator
from brain_memory.consolidation.llm import LLMConsolidator
from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.episodic.store import EpisodicStore
from brain_memory.models import ConsolidationReport, PatternProposal

logger = logging.getLogger(__name__)


class Consolidator:
    def __init__(
        self,
        db,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        config: MemoryConfig,
        embed_fn,
    ) -> None:
        self._db = db
        self._episodic = episodic_store
        self._semantic = semantic_store
        self._config = config
        self._embed_fn = embed_fn
        self._heuristic = HeuristicConsolidator()
        self._llm: LLMConsolidator | None = None

    def set_llm(self, llm: LLMConsolidator | None) -> None:
        """Engine wires the LLM consolidator in when configured."""
        self._llm = llm

    def consolidate(
        self,
        *,
        max_groups: int = 5,
        max_episodes_per_group: int = 20,
        min_support: int | None = None,
    ) -> ConsolidationReport:
        min_support = min_support or self._config.consolidation_min_support
        report = ConsolidationReport()
        now_iso = datetime.now(timezone.utc).isoformat()

        groups = candidate_groups(
            self._db,
            min_support=min_support,
            max_groups=max_groups,
            include_below_support=True,
        )
        report.groups_considered = sum(1 for g in groups if g.eligible)

        for group in groups:
            if not group.eligible:
                report.skipped.append(
                    f"{group.key}: below min_support ({group.episode_count}/{min_support})"
                )
                continue
            proposal, memory = self._consolidate_group(
                group,
                max_episodes_per_group=max_episodes_per_group,
                min_support=min_support,
                now_iso=now_iso,
            )
            if proposal is None or memory is None:
                report.skipped.append(
                    f"{group.key}: no proposal meeting support {min_support}"
                )
                continue
            self._db.upsert_consolidation_state(
                value=group.value,
                representative_kind=group.kind,
                last_consolidated_at=now_iso,
                episode_count=group.episode_count,
                semantic_id=memory.id,
            )
            (report.created if memory.version == 1 else report.updated).append(memory)
        return report

    # -- one group ----------------------------------------------------------------

    def _consolidate_group(
        self,
        group: CandidateGroup,
        *,
        max_episodes_per_group: int,
        min_support: int,
        now_iso: str,
    ):
        episode_ids = self._db.episode_ids_by_tag(group.kind, group.value)
        episodes = [e for e in self._episodic.get_many(episode_ids) if e.is_active]
        episodes.sort(key=lambda e: e.created_at, reverse=True)
        episodes = episodes[:max_episodes_per_group]

        proposal = self._propose(group.value, episodes, min_support)
        if proposal is None:
            return None, None

        embedding = self._embed_fn(proposal.statement)
        evidence = sorted({episodes[i].id for i in proposal.supporting_indexes})
        existing = self._semantic.get_by_concept(group.value)

        if existing is None:
            memory = self._semantic.create(
                concept=group.value,
                kind=proposal.kind,
                statement=proposal.statement,
                confidence=proposal.confidence,
                evidence_ids=evidence,
                embedding=embedding,
                change_reason="initial consolidation",
            )
        else:
            merged_evidence = sorted(set(existing.evidence_ids) | set(evidence))
            smoothed = round(0.5 * existing.confidence + 0.5 * proposal.confidence, 3)
            memory = self._semantic.update(
                existing,
                kind=proposal.kind,
                statement=proposal.statement,
                confidence=smoothed,
                evidence_ids=merged_evidence,
                embedding=embedding,
                change_reason=f"new evidence ({len(evidence)} supporting episodes)",
            )
        return proposal, memory

    def _propose(
        self, concept: str, episodes, min_support: int
    ) -> PatternProposal | None:
        if self._llm is not None:
            try:
                return self._llm.propose(concept, episodes, min_support=min_support)
            except Exception as exc:  # noqa: BLE001 — degrade to deterministic path
                logger.warning(
                    "LLM consolidation failed for '%s' (%s); using heuristic proposal",
                    concept,
                    exc,
                )
        return self._heuristic.propose(concept, episodes, min_support=min_support)
