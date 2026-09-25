"""Multi-factor scoring, normalization, and explanation.

Each factor is normalized to [0, 1] *before* weighting — raw quantities with
different scales (bm25 ranks, ages, access counts) are not comparable.  The
final score is the weighted sum divided by the total weight, so it stays in
[0, 1] even for non-normalized weight vectors.

Why explainable from day one: the explanations cost one dict comprehension
per candidate (the factors are computed anyway), and "why was this recalled"
is a first-class requirement of the design (doc §19), not a debugging tool
bolted on later.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from brain_memory.config import RetrievalWeights
from brain_memory.models import Episode, FactorScores

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize_bm25(rank: float) -> float:
    """Map FTS5 bm25 rank (0 = best, larger = worse) into (0, 1]."""
    return 1.0 / (1.0 + max(rank, 0.0))


def recency_factor(
    created_at: datetime,
    now: datetime | None = None,
    half_life_days: float = 14.0,
) -> float:
    """Exponential decay: an episode half as fresh every ``half_life_days``."""
    now = now or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / max(half_life_days, 1e-9))


def frequency_factor(access_count: int, cap: int = 50) -> float:
    """Log-scaled access frequency — raw counts would dominate the score."""
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / math.log1p(max(cap, 1)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_factors(
    *,
    episode: Episode,
    semantic: float,
    keyword: float,
    cue_entities: set[str],
    cue_context: set[str],
    now: datetime,
    half_life_days: float,
) -> FactorScores:
    """Normalize every factor for one candidate episode."""
    episode_entities = {e.casefold() for e in episode.entities}
    episode_context = {t.casefold() for t in episode.topics}
    if episode.context:
        episode_context |= _context_tokens(episode.context)
    return FactorScores(
        semantic=_clip01(semantic),
        keyword=_clip01(keyword),
        recency=recency_factor(episode.created_at, now, half_life_days),
        importance=_clip01(episode.importance),
        frequency=frequency_factor(episode.access_count),
        entity=jaccard(cue_entities, episode_entities),
        context=jaccard(cue_context, episode_context),
    )


def score(factors: FactorScores, weights: RetrievalWeights) -> float:
    weighted = (
        weights.semantic * factors.semantic
        + weights.keyword * factors.keyword
        + weights.recency * factors.recency
        + weights.importance * factors.importance
        + weights.frequency * factors.frequency
        + weights.entity * factors.entity
        + weights.context * factors.context
    )
    return weighted / weights.total()


def explain(factors: FactorScores, weights: RetrievalWeights) -> list[str]:
    """Human-readable contributions, most significant first."""
    total = weights.total()
    contributions = {
        "semantic": (weights.semantic, factors.semantic),
        "keyword": (weights.keyword, factors.keyword),
        "recency": (weights.recency, factors.recency),
        "importance": (weights.importance, factors.importance),
        "frequency": (weights.frequency, factors.frequency),
        "entity": (weights.entity, factors.entity),
        "context": (weights.context, factors.context),
    }
    ranked = sorted(
        (
            (name, weight, value, weight * value / total)
            for name, (weight, value) in contributions.items()
            if weight > 0 and value > 0
        ),
        key=lambda item: -item[3],
    )
    return [
        f"{name}: {value:.2f} × w{weight:.2f} → {contribution:.2f}"
        for name, weight, value, contribution in ranked
    ]


def _context_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _LATIN_RE.findall(text.lower()):
        if len(token) >= 2:
            tokens.add(token)
    cjk = _CJK_RE.findall(text)
    tokens.update(f"{a}{b}" for a, b in zip(cjk, cjk[1:]))
    return tokens


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
