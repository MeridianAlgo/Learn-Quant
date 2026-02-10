import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Correlation Analysis"))

from correlation_analysis import (
    correlation_matrix,
    correlation_stability,
    ewma_correlation,
    pearson_correlation,
    rolling_correlation,
    tail_correlation,
)


def test_pearson_correlation():
    returns1 = [0.01, -0.02, 0.015, -0.01, 0.02]
    returns2 = [0.008, -0.015, 0.012, -0.008, 0.018]
    corr = pearson_correlation(returns1, returns2)
    assert isinstance(corr, float)
    assert -1 <= corr <= 1


def test_rolling_correlation():
    returns1 = [0.01, -0.02, 0.015, -0.01, 0.02, 0.005, -0.008]
    returns2 = [0.008, -0.015, 0.012, -0.008, 0.018, 0.004, -0.006]
    rolling = rolling_correlation(returns1, returns2, window=3)
    assert len(rolling) == len(returns1) - 3 + 1
    assert all(-1 <= c <= 1 for c in rolling)


def test_correlation_matrix():
    returns_dict = {
        "asset1": [0.01, -0.02, 0.015, -0.01, 0.02],
        "asset2": [0.008, -0.015, 0.012, -0.008, 0.018],
        "asset3": [0.012, -0.018, 0.010, -0.012, 0.022],
    }
    corr_matrix, assets = correlation_matrix(returns_dict)
    assert corr_matrix.shape == (3, 3)
    assert len(assets) == 3
    assert np.allclose(np.diag(corr_matrix), 1.0)


def test_ewma_correlation():
    returns1 = [0.01, -0.02, 0.015, -0.01, 0.02, 0.005, -0.008]
    returns2 = [0.008, -0.015, 0.012, -0.008, 0.018, 0.004, -0.006]
    ewma_corr = ewma_correlation(returns1, returns2)
    assert len(ewma_corr) == len(returns1)
    assert all(-1 <= c <= 1 for c in ewma_corr)


def test_tail_correlation():
    np.random.seed(42)
    returns1 = np.random.normal(0, 0.02, 100).tolist()
    returns2 = np.random.normal(0, 0.02, 100).tolist()
    lower_tail, upper_tail = tail_correlation(returns1, returns2)
    assert isinstance(lower_tail, (float, type(np.nan)))
    assert isinstance(upper_tail, (float, type(np.nan)))


def test_correlation_stability():
    np.random.seed(42)
    returns1 = np.random.normal(0, 0.02, 100).tolist()
    returns2 = (0.5 * np.array(returns1) + 0.5 * np.random.normal(0, 0.02, 100)).tolist()
    stability = correlation_stability(returns1, returns2, n_splits=5)
    assert "mean" in stability
    assert "std" in stability
    assert "min" in stability
    assert "max" in stability
    assert "range" in stability
