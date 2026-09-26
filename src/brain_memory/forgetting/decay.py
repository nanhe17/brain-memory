"""记忆强度与衰减扫描。

两个机制，各司一职：

* **active -> archived** 由强度驱动：记忆的计算强度跌破
  ``decay_archive_threshold`` 就不再值得留在活跃集。重要性与置信度
  构成*静态下限*，因此被显式标记重要的记忆永远不会衰减进归档——
  恰好是文档的"强记忆存续"。
* **archived -> forgotten** 由驻留时间驱动：归档的记忆无法被召回，
  last_touched 锚点冻结，超过 ``decay_forget_after_days`` 天后落入
  终态（仍是软的）。第二个强度阈值在这里行不通：静态下限会让大多数
  记忆永远高居任何更低阈值之上。

证据保护与巩固（Phase 3）闭环：被*活跃*语义记忆引用的 episode 豁免
扫描——活着的关键住它的证据；知识本身衰减后，证据才被释放去自然老化。
"""

from __future__ import annotations

from datetime import datetime, timezone

from brain_memory.config import MemoryConfig
from brain_memory.episodic.store import EpisodicStore
from brain_memory.models import DecayReport, SemanticMemory
from brain_memory.retrieval import ranking as _rh
from brain_memory.storage.db import Database

# 强度权重是校准单元，刻意不做用户配置（见 DESIGN.md）。
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
    """新近度锚点：一条记忆的"新鲜度"以其最近一次使用为准。"""
    if last_accessed is not None and last_accessed > created_at:
        return last_accessed
    return created_at


def episode_strength(
    episode, *, now: datetime, half_life_days: float
) -> float:
    """情景记忆强度：重要性/新近度/频率/置信度的加权混合。"""
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
    """语义记忆强度：证据越多越持久（互补学习系统风格）。"""
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
    """补齐 naive datetime 的 UTC 时区。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class DecaySweeper:
    """遗忘扫描器：强度归档 + 驻留遗忘 + 证据保护。"""

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
        """全量扫描：计算强度、应用顺序迁移、返回报告。

        ``dry_run=True`` 只算不改；``now`` 允许时间推演。
        """
        now = _aware(now) if now is not None else datetime.now(timezone.utc)
        half_life = self._config.recency_half_life_days
        archive_threshold = self._config.decay_archive_threshold
        forget_after_days = self._config.decay_forget_after_days
        report = DecayReport(dry_run=dry_run)

        # 活跃知识的证据豁免扫描
        protected: set[int] = set()
        for memory in self._semantic.list_active():
            protected.update(memory.evidence_ids)

        # 第一段：active -> archived（强度阈值）
        for episode in self._episodic.all_active():
            report.swept_episodes += 1
            if episode.id in protected:
                report.protected_evidence_count += 1
                continue
            strength = episode_strength(episode, now=now, half_life_days=half_life)
            if strength < archive_threshold:
                report.archived_episode_ids.append(episode.id)

        # 第二段：archived -> forgotten（驻留时长；归档期间 last_touched 冻结）
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

        # 应用迁移（dry_run 时跳过）
        for episode_id in report.archived_episode_ids:
            self._episodic.archive(episode_id)
        for episode_id in report.forgotten_episode_ids:
            self._episodic.forget(episode_id)
        for semantic_id in report.archived_semantic_ids:
            self._semantic.archive(semantic_id)
        for semantic_id in report.forgotten_semantic_ids:
            self._semantic.forget(semantic_id)
        return report
