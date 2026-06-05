import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Extreme Value Theory"))
from extreme_value_theory import (
    fit_gpd_mom,
    hill_estimator,
    mean_excess,
    pot_var_es,
)


def test_fit_gpd_returns_params():
    rng = np.random.default_rng(0)
    # Exponential excesses -> GPD with xi ~ 0.
    excess = rng.exponential(1.0, 2000)
    fit = fit_gpd_mom(excess)
    assert "xi" in fit and "beta" in fit
    assert abs(fit["xi"]) < 0.15
    assert fit["beta"] > 0


def test_pot_var_es_ordering():
    rng = np.random.default_rng(1)
    losses = -(0.0005 + 0.012 * rng.standard_t(4, size=4000))
    evt = pot_var_es(losses, confidence=0.99, threshold_quantile=0.90)
    # ES is always at least as large as VaR in the tail.
    assert evt["es"] >= evt["var"]
    assert evt["var"] > 0
    assert evt["n_excess"] > 0


def test_higher_confidence_higher_var():
    rng = np.random.default_rng(2)
    losses = -(0.0005 + 0.012 * rng.standard_t(5, size=4000))
    v95 = pot_var_es(losses, confidence=0.95)["var"]
    v99 = pot_var_es(losses, confidence=0.99)["var"]
    assert v99 > v95


def test_hill_estimator_positive_for_heavy_tail():
    rng = np.random.default_rng(3)
    # Pareto data has a positive tail index.
    data = (1 - rng.random(5000)) ** (-1.0 / 3.0)  # Pareto, alpha=3 -> xi~1/3
    xi = hill_estimator(data, k=300)
    assert 0.1 < xi < 0.8


def test_mean_excess_shape():
    rng = np.random.default_rng(4)
    losses = rng.exponential(1.0, 1000)
    thr = np.linspace(0.5, 2.0, 5)
    me = mean_excess(losses, thr)
    assert me.shape == (5,)


def test_small_sample_raises():
    try:
        pot_var_es([0.1, 0.2, 0.3], confidence=0.99)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
