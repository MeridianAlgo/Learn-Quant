import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Expected Shortfall"))
from expected_shortfall import cornish_fisher_es, es_summary, historical_es, parametric_es


@pytest.fixture
def returns():
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 500)


def test_historical_es_positive(returns):
    es = historical_es(returns, 0.95)
    assert isinstance(es, float)
    assert es > 0


def test_parametric_es_positive(returns):
    es = parametric_es(returns, 0.95)
    assert isinstance(es, float)
    assert es > 0


def test_cornish_fisher_es_positive(returns):
    es = cornish_fisher_es(returns, 0.95)
    assert isinstance(es, float)
    assert es > 0


def test_es_greater_than_var(returns):
    alpha = 0.05
    var = -np.percentile(returns, alpha * 100)
    es = historical_es(returns, 0.95)
    assert es >= var


def test_es_increases_with_confidence(returns):
    es_90 = historical_es(returns, 0.90)
    es_95 = historical_es(returns, 0.95)
    es_99 = historical_es(returns, 0.99)
    assert es_90 <= es_95 <= es_99


def test_es_summary_keys(returns):
    summary = es_summary(returns)
    assert "historical_es" in summary
    assert "parametric_es" in summary
    assert "cornish_fisher_es" in summary
    assert "confidence_level" in summary


def test_es_summary_confidence_stored(returns):
    summary = es_summary(returns, confidence_level=0.99)
    assert summary["confidence_level"] == 0.99


def test_all_positive_returns_low_es():
    positive = np.full(100, 0.01)
    es = historical_es(positive, 0.95)
    assert es < 0  # ES is negative when all returns are positive (no loss)
