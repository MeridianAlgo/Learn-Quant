import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Quantitative Methods - Regime Detection"))
from regime_detection import moving_average_regime, regime_stats, volatility_regime


@pytest.fixture
def trending_prices():
    np.random.seed(42)
    return 100 + np.cumsum(np.random.normal(0.1, 1.0, 300))


@pytest.fixture
def mixed_returns():
    np.random.seed(42)
    bull = np.random.normal(0.001, 0.01, 200)
    bear = np.random.normal(-0.002, 0.02, 200)
    return np.concatenate([bull, bear])


def test_ma_regime_shape(trending_prices):
    labels = moving_average_regime(trending_prices, short_window=20, long_window=50)
    assert len(labels) == len(trending_prices)


def test_ma_regime_values(trending_prices):
    labels = moving_average_regime(trending_prices, short_window=20, long_window=50)
    valid = labels[~np.isnan(labels)]
    assert set(valid).issubset({0.0, 1.0})


def test_ma_regime_warmup_nan(trending_prices):
    labels = moving_average_regime(trending_prices, short_window=20, long_window=50)
    assert np.all(np.isnan(labels[:49]))


def test_vol_regime_shape(mixed_returns):
    labels = volatility_regime(mixed_returns, window=21, n_regimes=3)
    assert len(labels) == len(mixed_returns)


def test_vol_regime_values(mixed_returns):
    labels = volatility_regime(mixed_returns, window=21, n_regimes=3)
    valid = labels[~np.isnan(labels)]
    assert set(valid).issubset({0.0, 1.0, 2.0})


def test_regime_stats_keys(mixed_returns):
    labels = volatility_regime(mixed_returns, window=21, n_regimes=2)
    valid_labels = labels[~np.isnan(labels)]
    stats = regime_stats(mixed_returns[~np.isnan(labels)], valid_labels)
    for r_stats in stats.values():
        assert "mean" in r_stats
        assert "std" in r_stats
        assert "count" in r_stats


def test_gmm_regime_optional():
    """GMM is optional (requires sklearn); verify graceful fallback on ImportError."""
    try:
        from regime_detection import gaussian_mixture_regime
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 200)
        result = gaussian_mixture_regime(returns, n_regimes=2)
        assert "labels" in result
        assert len(result["labels"]) == 200
    except ImportError:
        pytest.skip("scikit-learn not installed")
