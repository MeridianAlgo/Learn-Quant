import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Markov Chains"))
from markov_chains import (
    expected_return_time,
    is_stochastic,
    n_step,
    normalize_rows,
    simulate,
    stationary_distribution,
)

P = np.array([[0.90, 0.03, 0.07], [0.05, 0.85, 0.10], [0.15, 0.10, 0.75]])


def test_normalize_rows_sums_to_one():
    counts = np.array([[820, 30, 60], [25, 300, 40], [70, 45, 410]])
    Pn = normalize_rows(counts)
    assert np.allclose(Pn.sum(axis=1), 1.0)


def test_normalize_zero_row_is_uniform():
    Pn = normalize_rows(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert np.allclose(Pn[0], [0.5, 0.5])


def test_normalize_rejects_negatives():
    try:
        normalize_rows(np.array([[-1.0, 2.0]]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_is_stochastic():
    assert is_stochastic(P)
    assert not is_stochastic(np.array([[0.5, 0.4], [0.1, 0.9]]))


def test_n_step_is_stochastic():
    P5 = n_step(P, 5)
    assert np.allclose(P5.sum(axis=1), 1.0)


def test_n_step_zero_is_identity():
    assert np.allclose(n_step(P, 0), np.eye(3))


def test_stationary_is_fixed_point():
    pi = stationary_distribution(P)
    assert np.allclose(pi @ P, pi, atol=1e-9)
    assert abs(pi.sum() - 1.0) < 1e-12
    assert np.all(pi >= 0)


def test_stationary_matches_long_run_powers():
    # P^n rows all converge to the stationary distribution.
    pi = stationary_distribution(P)
    big = n_step(P, 500)
    assert np.allclose(big[0], pi, atol=1e-6)


def test_expected_return_time_is_reciprocal():
    pi = stationary_distribution(P)
    assert abs(expected_return_time(P, 0) - 1.0 / pi[0]) < 1e-9


def test_simulate_shape_and_range():
    path = simulate(P, start=0, steps=50, random_state=7)
    assert path.shape == (51,)
    assert path[0] == 0
    assert set(np.unique(path)).issubset({0, 1, 2})


def test_simulate_empirical_matches_stationary():
    # A long path should visit states roughly in stationary proportions.
    path = simulate(P, start=0, steps=20000, random_state=0)
    freq = np.bincount(path, minlength=3) / len(path)
    assert np.allclose(freq, stationary_distribution(P), atol=0.03)
