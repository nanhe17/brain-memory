"""Memory strength and the decay sweep.

Two mechanisms, each with one job:

* **active -> archived** is strength-driven: a memory whose computed strength
  falls below ``decay_archive_threshold`` is no longer worth keeping in the
  live set.  Importance and confidence form a *static floor*, so explicitly
  important memories never decay into the archive — exactly the doc's
  "strong memories persist".
* **archived -> forgotten** is dwell-driven: an archived memory cannot be
  recalled, its last_touched anchor freezes, and after
  ``decay_forget_after_days`` it falls into the terminal (still soft)
  FORGOTTEN state.  A second strength threshold would not work here: the
  static floor would keep most memories above any lower threshold forever.

Evidence protection closes the loop with consolidation (Phase 3): episodes
cited by an *active* semantic memory are exempt from the sweep — knowledge
that is alive keeps its evidence alive, and only when the knowledge itself
decays is the evidence released to age naturally.
"""

from __future__ import annotations

from datetime import datetime, timezone

from brain_memory.config import MemoryConfig
from brain_memory.episodic.store import EpisodicStore
from brain_memory.models import DecayReport, Episode, SemanticMemory
from brain_memory.retrieval import ranking as _rh
from brain_memory.storage.db import Database

# Calibration units, deliberately not user configuration (see DESIGN.md).
EPISODE_STRENGTH_WEIGHTS = {
    "importance": 0.30,
    "recency": 0.40,
    "frequency": 0.10,
    "confidence": 0.20,
}
SEMANTIC_STRENGTH_WEIGHTS = {
    "confidence": 0.25,
    "recency": 0.35,
    "frequency": 0.10,
    "evidence_support": 0.30,
}
_EVIDENCE_FULL_SUPPORT = 5


def last_touched(created_at: datetime, last_accessed: datetime | None) -> datetime:
    """The recency anchor: a memory is as fresh as its last use."""
    if last_accessed is not None and last_accessed > created_at:
        return last_accessed
    return created_at


def episode_strength(
    episode: Episode, *, now: datetime, half_life_days: float
) -> float:
    recency = _rh.recency_factor(
        last_touched(episode.created_at, episode.last_accessed), now, half_life_days
    )
    frequency = _rh.frequency_factor(episode.access_count)
    w = EPISODE_STRENGTH_WEIGHTS
    return round(
        w["importance"] * episode.importance
        + w["recency"] * recency
        + w["frequency"] * frequency
        + w["confidence"] * episode.confidence,
        4,
    )


def semantic_strength(
    memory: SemanticMemory, *, now: datetime, half_life_days: float
) -> float:
    recency = _rh.recency_factor(
        last_touched(memory.updated_at, memory.last_accessed), now, half_life_days
    )
    frequency = _rh.frequency_factor(memory.access_count)
    evidence_support = min(1.0, len(memory.evidence_ids) / _EVIDENCE_FULL_SUPPORT)
    w = SEMANTIC_STRENGTH_WEIGHTS
    return round(
        w["confidence"] * memory.confidence
        + w["recency"] * recency
        + w["frequency"] * frequency
        + w["evidence_support"] * evidence_support,
        4,
    )


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class DecaySweeper:
    def __init__(
        self,
        db: Database,
        episodic_store: EpisodicStore,
        semantic_store,
        config: MemoryConfig,
    ) -> None:
        self._db = db
        self._episodic = episodic_store
        self._semantic = semantic_store
        self._config = config

    def sweep(self, *, dry_run: bool = False, now: datetime | None = None) -> DecayReport:
        now = _aware(now) if now is not None else datetime.now(timezone.utc)
        half_life = self._config.recency_half_life_days
        archive_threshold = self._config.decay_archive_threshold
        forget_after_days = self._config.decay_forget_after_days
        report = DecayReport(dry_run=dry_run)

        # evidence of living knowledge is exempt
        protected: set[int] = set()
        for memory in self._semantic.list_active():
            protected.update(memory.evidence_ids)

        for episode in self._episodic.all_active():
            report.swept_episodes += 1
            if episode.id in protected:
                report.protected_evidence_count += 1
                continue
            strength = episode_strength(episode, now=now, half_life_days=half_life)
            if strength < archive_threshold:
                report.archived_episode_ids.append(episode.id)

        # archived -> forgotten by dwell time (last_touched freezes in archive)
        for row in self._db.list_archived_episodes():
            touched = last_touched(
                self._db.parse_dt(row["created_at"]),
                self._db.parse_dt(row["last_accessed"]),
            )
            if (now - _aware(touched)).total_seconds() / 86400.0 > forget_after_days:
                report.forgotten_episode_ids.append(row["id"])

        for memory in self._semantic.list_active():
            report.swept_semantics += 1
            strength = semantic_strength(memory, now=now, half_life_days=half_life)
            if strength < archive_threshold:
                report.archived_semantic_ids.append(memory.id)

        for row in self._db.list_archived_semantics():
            touched = last_touched(
                self._db.parse_dt(row["updated_at"]),
                self._db.parse_dt(row["last_accessed"]),
            )
            if (now - _aware(touched)).total_seconds() / 86400.0 > forget_after_days:
                report.forgotten_semantic_ids.append(row["id"])

        if dry_run:
            return report

        for episode_id in report.archived_episode_ids:
            self._episodic.archive(episode_id)
        for episode_id in report.forgotten_episode_ids:
            self._episodic.forget(episode_id)
        for semantic_id in report.archived_semantic_ids:
            self._semantic.archive(semantic_id)
        for semantic_id in report.forgotten_semantic_ids:
            self._semantic.forget(semantic_id)
        return report
