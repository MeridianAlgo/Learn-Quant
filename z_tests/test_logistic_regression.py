import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - Logistic Regression"))
from logistic_regression import (
    accuracy,
    fit,
    log_loss,
    predict,
    predict_proba,
    sigmoid,
    standardize,
)


def _two_blobs(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-1.5, -1.5], 1.0, size=(n, 2))
    X1 = rng.normal([1.5, 1.5], 1.0, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


def test_sigmoid_range_and_midpoint():
    assert abs(sigmoid(0.0) - 0.5) < 1e-12
    vals = sigmoid(np.array([-50.0, 0.0, 50.0]))
    assert np.all(vals >= 0) and np.all(vals <= 1)


def test_sigmoid_stable_for_large_negative():
    # Naive exp(-z) would overflow; stable version must not produce nan/inf.
    out = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] < 1e-9 and out[1] > 1.0 - 1e-9


def test_log_loss_zero_when_perfect():
    y = np.array([0.0, 1.0, 1.0])
    p = np.array([0.0, 1.0, 1.0])
    assert log_loss(y, p) < 1e-10


def test_log_loss_penalises_confident_wrong():
    assert log_loss(np.array([1.0]), np.array([0.01])) > log_loss(np.array([1.0]), np.array([0.4]))


def test_fit_separates_blobs():
    X, y = _two_blobs()
    Xs, _, _ = standardize(X)
    w, b = fit(Xs, y, lr=0.5, epochs=2000, l2=0.01)
    assert accuracy(y, predict(Xs, w, b)) > 0.95


def test_predict_proba_in_unit_interval():
    X, y = _two_blobs()
    Xs, _, _ = standardize(X)
    w, b = fit(Xs, y, lr=0.5, epochs=500)
    p = predict_proba(Xs, w, b)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_standardize_zero_mean_unit_std():
    X, _ = _two_blobs()
    Xs, mu, sd = standardize(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-9)


def test_standardize_constant_column_no_nan():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Xs, _, _ = standardize(X)
    assert np.all(np.isfinite(Xs))


def test_loss_decreases_with_training():
    X, y = _two_blobs()
    Xs, _, _ = standardize(X)
    w0, b0 = fit(Xs, y, lr=0.5, epochs=5)
    w1, b1 = fit(Xs, y, lr=0.5, epochs=1000)
    assert log_loss(y, predict_proba(Xs, w1, b1)) < log_loss(y, predict_proba(Xs, w0, b0))
