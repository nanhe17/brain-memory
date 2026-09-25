from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from brain_memory.retrieval import ranking


def test_normalize_bm25_bounds_and_monotonicity():
    assert ranking.normalize_bm25(0.0) == 1.0
    best = ranking.normalize_bm25(0.5)
    worse = ranking.normalize_bm25(4.0)
    assert 0.0 < worse < best < 1.0


def test_recency_half_life():
    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    created = now - timedelta(days=14)
    assert ranking.recency_factor(created, now, half_life_days=14.0) == pytest.approx(0.5)
    fresh = ranking.recency_factor(now, now)
    assert fresh == pytest.approx(1.0)
    old = ranking.recency_factor(now - timedelta(days=140), now, half_life_days=14.0)
    assert old < 0.001


def test_frequency_log_scale_and_cap():
    assert ranking.frequency_factor(0) == 0.0
    one = ranking.frequency_factor(1)
    ten = ranking.frequency_factor(10)
    hundred = ranking.frequency_factor(100)
    assert 0 < one < ten < 1.0
    assert hundred == 1.0  # capped


def test_score_stays_in_unit_interval_with_unnormalized_weights():
    from brain_memory.config import RetrievalWeights
    from brain_memory.models import FactorScores

    weights = RetrievalWeights(semantic=2.0, keyword=2.0, recency=2.0, importance=2.0,
                               frequency=2.0, entity=2.0, context=2.0)
    factors = FactorScores(semantic=1.0, keyword=1.0, recency=1.0, importance=1.0,
                           frequency=1.0, entity=1.0, context=1.0)
    assert ranking.score(factors, weights) == pytest.approx(1.0)


def test_explain_sorted_by_contribution():
    from brain_memory.config import RetrievalWeights
    from brain_memory.models import FactorScores

    weights = RetrievalWeights()
    factors = FactorScores(semantic=0.9, keyword=0.0, recency=0.5, importance=0.2,
                           frequency=0.0, entity=0.1, context=0.0)
    reasons = ranking.explain(factors, weights)
    assert reasons[0].startswith("semantic")
    assert all("keyword" not in line and "frequency" not in line for line in reasons)
    contributions = [float(line.rsplit("→", 1)[1]) for line in reasons]
    assert contributions == sorted(contributions, reverse=True)
