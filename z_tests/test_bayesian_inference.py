import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Bayesian Inference"))
from bayesian_inference import (
    beta_binomial_update,
    beta_credible_interval,
    beta_mean,
    normal_known_variance_update,
    probability_greater_than,
)


def test_beta_update_adds_counts():
    a, b = beta_binomial_update(1, 1, successes=7, failures=3)
    assert a == 8 and b == 4


def test_beta_mean_value():
    a, b = beta_binomial_update(1, 1, 7, 3)
    assert abs(beta_mean(a, b) - 8 / 12) < 1e-12


def test_beta_update_rejects_bad_prior():
    try:
        beta_binomial_update(0, 1, 1, 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_beta_update_rejects_negative_counts():
    try:
        beta_binomial_update(1, 1, -1, 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_credible_interval_brackets_mean():
    a, b = beta_binomial_update(1, 1, 7, 3)
    lo, hi = beta_credible_interval(a, b, 0.95)
    assert 0 <= lo < beta_mean(a, b) < hi <= 1


def test_credible_interval_narrows_with_data():
    a1, b1 = beta_binomial_update(1, 1, 7, 3)
    a2, b2 = beta_binomial_update(1, 1, 700, 300)
    lo1, hi1 = beta_credible_interval(a1, b1, 0.95)
    lo2, hi2 = beta_credible_interval(a2, b2, 0.95)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_credible_interval_bad_level():
    try:
        beta_credible_interval(2, 2, 1.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_probability_greater_than_symmetric_prior():
    # Beta(2, 2) is symmetric about 0.5.
    assert abs(probability_greater_than(2, 2, 0.5) - 0.5) < 1e-9


def test_normal_update_shrinks_toward_prior():
    samples = [0.05, 0.05, 0.05]
    post_mean, post_var = normal_known_variance_update(0.0, 0.01, samples, 0.04)
    # Posterior sits between the prior (0) and the sample mean (0.05).
    assert 0.0 < post_mean < 0.05
    assert post_var < 0.01  # variance always decreases with data


def test_normal_update_empty_data_returns_prior():
    m, v = normal_known_variance_update(0.1, 0.02, [], 0.04)
    assert m == 0.1 and v == 0.02


def test_normal_update_more_data_moves_toward_sample():
    few = normal_known_variance_update(0.0, 0.01, [0.05] * 2, 0.04)[0]
    many = normal_known_variance_update(0.0, 0.01, [0.05] * 50, 0.04)[0]
    assert many > few  # more data -> closer to sample mean of 0.05
