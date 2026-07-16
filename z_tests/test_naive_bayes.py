import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - Naive Bayes"))
from naive_bayes import accuracy, fit_gaussian_nb, predict, predict_proba


def make_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n)])
    b = np.column_stack([rng.normal(4, 1, n), rng.normal(4, 1, n)])
    X = np.vstack([a, b])
    y = np.array(["a"] * n + ["b"] * n)
    return X, y


def test_separable_classes_high_accuracy():
    X, y = make_data()
    model = fit_gaussian_nb(X, y)
    assert accuracy(model, X, y) > 0.98


def test_predict_obvious_points():
    X, y = make_data()
    model = fit_gaussian_nb(X, y)
    assert predict(model, np.array([[0.0, 0.0]]))[0] == "a"
    assert predict(model, np.array([[4.0, 4.0]]))[0] == "b"


def test_proba_rows_sum_to_one():
    X, y = make_data(seed=1)
    model = fit_gaussian_nb(X, y)
    probs = predict_proba(model, X[:20])
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0)


def test_proba_confident_far_from_boundary():
    X, y = make_data(seed=2)
    model = fit_gaussian_nb(X, y)
    probs = predict_proba(model, np.array([[-1.0, -1.0]]))[0]
    a_index = int(np.where(model["classes"] == "a")[0][0])
    assert probs[a_index] > 0.99


def test_priors_reflect_class_frequencies():
    rng = np.random.default_rng(3)
    X = np.vstack(
        [
            np.column_stack([rng.normal(0, 1, 300), rng.normal(0, 1, 300)]),
            np.column_stack([rng.normal(5, 1, 100), rng.normal(5, 1, 100)]),
        ]
    )
    y = np.array(["common"] * 300 + ["rare"] * 100)
    model = fit_gaussian_nb(X, y)
    priors = np.exp(model["log_priors"])
    assert abs(priors.sum() - 1.0) < 1e-12
    common_index = int(np.where(model["classes"] == "common")[0][0])
    assert abs(priors[common_index] - 0.75) < 1e-12


def test_constant_feature_does_not_crash():
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.normal(0, 1, 100), np.ones(100)])
    X[50:, 0] += 5
    y = np.array(["a"] * 50 + ["b"] * 50)
    model = fit_gaussian_nb(X, y)
    assert accuracy(model, X, y) > 0.9


def test_three_classes():
    rng = np.random.default_rng(5)
    X = np.vstack(
        [
            rng.normal(0, 0.5, (100, 2)),
            rng.normal(3, 0.5, (100, 2)),
            rng.normal(6, 0.5, (100, 2)),
        ]
    )
    y = np.array(["low"] * 100 + ["mid"] * 100 + ["high"] * 100)
    model = fit_gaussian_nb(X, y)
    assert accuracy(model, X, y) > 0.95


def test_single_class_raises():
    X = np.ones((10, 2))
    y = np.array(["a"] * 10)
    with pytest.raises(ValueError):
        fit_gaussian_nb(X, y)


def test_mismatched_rows_raise():
    with pytest.raises(ValueError):
        fit_gaussian_nb(np.ones((5, 2)), np.array(["a", "b"]))


def test_one_dimensional_features_raise():
    with pytest.raises(ValueError):
        fit_gaussian_nb(np.ones(5), np.array(["a"] * 5))
