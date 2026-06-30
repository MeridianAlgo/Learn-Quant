import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - Gradient Descent"))
from gradient_descent import (
    gradient_descent,
    mse,
    predict,
    sgd,
    standardize,
)


def _make_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_w = np.array([3.0, -2.0])
    y = X @ true_w + 5.0 + rng.normal(scale=0.1, size=n)
    return X, y, true_w, 5.0


def test_predict_and_mse_basics():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = np.array([2.0, 3.0])
    assert list(predict(X, w, 1.0)) == [3.0, 4.0]
    assert mse([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert mse([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_standardize_zero_mean_unit_std():
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    Xs, mean, std = standardize(X)
    assert np.allclose(Xs.mean(axis=0), 0.0)
    assert np.allclose(Xs.std(axis=0), 1.0)
    assert np.allclose(mean, [2.0, 200.0])


def test_standardize_constant_column_does_not_crash():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Xs, _, _ = standardize(X)
    assert np.all(np.isfinite(Xs))


def test_batch_descent_loss_decreases():
    X, y, _, _ = _make_data()
    Xs, _, _ = standardize(X)
    res = gradient_descent(Xs, y, lr=0.1, epochs=200)
    assert res["history"][-1] < res["history"][0]
    assert res["history"][-1] < 0.05


def test_batch_descent_recovers_coefficients():
    X, y, true_w, true_b = _make_data()
    Xs, mean, std = standardize(X)
    res = gradient_descent(Xs, y, lr=0.1, epochs=500)
    recovered = res["weights"] / std  # undo the scaling
    assert np.allclose(recovered, true_w, atol=0.1)


def test_sgd_also_converges():
    X, y, true_w, _ = _make_data()
    Xs, mean, std = standardize(X)
    res = sgd(Xs, y, lr=0.02, epochs=60, seed=1)
    recovered = res["weights"] / std
    assert np.allclose(recovered, true_w, atol=0.2)


def test_sgd_is_deterministic_with_seed():
    X, y, _, _ = _make_data()
    Xs, _, _ = standardize(X)
    a = sgd(Xs, y, lr=0.02, epochs=10, seed=42)
    b = sgd(Xs, y, lr=0.02, epochs=10, seed=42)
    assert np.allclose(a["weights"], b["weights"])
    assert a["bias"] == pytest.approx(b["bias"])


def test_too_large_learning_rate_diverges():
    X, y, _, _ = _make_data()
    Xs, _, _ = standardize(X)
    res = gradient_descent(Xs, y, lr=5.0, epochs=100)
    # An unstable rate should not settle at a small loss (it blows up to a huge
    # value or to inf/nan). Phrased so both cases pass, since nan < 1.0 is False.
    assert not (res["history"][-1] < 1.0)
