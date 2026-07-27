import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - K Nearest Neighbors"))
from knn import KNearestNeighbors, accuracy, standardize


def separable_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    up = rng.normal([2.0, 2.0], 0.5, size=(n, 2))
    down = rng.normal([-2.0, -2.0], 0.5, size=(n, 2))
    x = np.vstack([up, down])
    y = np.array([1] * n + [0] * n)
    return x, y


def test_standardize_gives_zero_mean_unit_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(50, 10, size=(200, 3))
    scaled, mean, scale = standardize(x)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-12)
    assert np.allclose((x - mean) / scale, scaled)


def test_standardize_survives_a_constant_column():
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    scaled, _, scale = standardize(x)
    assert scale[0] == 1.0
    assert np.all(np.isfinite(scaled))


def test_k_of_one_reproduces_training_labels():
    x, y = separable_data(seed=2)
    model = KNearestNeighbors(k=1).fit(x, y)
    assert accuracy(y, model.predict(x)) == 1.0


def test_separable_clouds_are_classified():
    x, y = separable_data(seed=3)
    model = KNearestNeighbors(k=5).fit(x[:300], y[:300])
    assert accuracy(y[300:], model.predict(x[300:])) > 0.95


def test_prediction_follows_the_nearest_cloud():
    x, y = separable_data(seed=4)
    model = KNearestNeighbors(k=5).fit(x, y)
    assert model.predict([[3.0, 3.0]])[0] == 1
    assert model.predict([[-3.0, -3.0]])[0] == 0


def test_probabilities_sum_to_one():
    x, y = separable_data(seed=5)
    model = KNearestNeighbors(k=7).fit(x, y)
    probas = model.predict_proba([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]])
    assert np.allclose(probas.sum(axis=1), 1.0)
    assert probas.shape == (3, 2)


def test_probability_is_confident_deep_inside_a_cloud():
    x, y = separable_data(seed=6)
    model = KNearestNeighbors(k=5).fit(x, y)
    assert model.predict_proba([[5.0, 5.0]])[0][1] == 1.0


def test_regression_averages_the_neighbors():
    x = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]
    y = [1.0, 2.0, 3.0, 100.0, 101.0, 102.0]
    model = KNearestNeighbors(k=3, task="regression").fit(x, y)
    assert abs(model.predict([[1.0]])[0] - 2.0) < 1e-12
    assert abs(model.predict([[11.0]])[0] - 101.0) < 1e-12


def test_regression_recovers_a_smooth_function():
    rng = np.random.default_rng(7)
    grid = np.linspace(0, 10, 400).reshape(-1, 1)
    values = np.sin(grid.ravel())
    model = KNearestNeighbors(k=5, task="regression").fit(grid, values)
    probe = rng.uniform(1, 9, size=(50, 1))
    predicted = model.predict(probe)
    assert np.max(np.abs(predicted - np.sin(probe.ravel()))) < 0.1


def test_distance_weighting_favours_the_closer_neighbor():
    x = [[0.0], [1.0], [5.0]]
    y = [0.0, 0.0, 10.0]
    uniform = KNearestNeighbors(k=3, task="regression").fit(x, y)
    weighted = KNearestNeighbors(k=3, task="regression", weights="distance").fit(x, y)
    assert weighted.predict([[0.5]])[0] < uniform.predict([[0.5]])[0]


def test_exact_hit_owns_the_distance_weighted_vote():
    x = [[0.0], [1.0], [2.0]]
    y = [5.0, 50.0, 500.0]
    model = KNearestNeighbors(k=3, task="regression", weights="distance").fit(x, y)
    assert abs(model.predict([[1.0]])[0] - 50.0) < 1e-12


def test_large_k_smooths_toward_the_overall_mean():
    x, y = separable_data(n=100, seed=8)
    model = KNearestNeighbors(k=200, task="regression").fit(x, y.astype(float))
    assert abs(model.predict([[5.0, 5.0]])[0] - 0.5) < 0.05


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        KNearestNeighbors().predict([[1.0, 2.0]])


def test_wrong_feature_count_raises():
    x, y = separable_data(n=20, seed=9)
    model = KNearestNeighbors(k=3).fit(x, y)
    with pytest.raises(ValueError):
        model.predict([[1.0]])


def test_mismatched_rows_raise():
    with pytest.raises(ValueError):
        KNearestNeighbors(k=1).fit([[1.0], [2.0]], [0])


def test_k_larger_than_training_set_raises():
    with pytest.raises(ValueError):
        KNearestNeighbors(k=10).fit([[1.0], [2.0]], [0, 1])


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        KNearestNeighbors(k=0)
    with pytest.raises(ValueError):
        KNearestNeighbors(task="clustering")
    with pytest.raises(ValueError):
        KNearestNeighbors(weights="gaussian")


def test_predict_proba_rejected_for_regression():
    model = KNearestNeighbors(k=1, task="regression").fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError):
        model.predict_proba([[0.5]])
