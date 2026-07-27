import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Entropy"))
from entropy import (
    discretize,
    entropy_from_samples,
    kl_divergence,
    mutual_information,
    normalized_entropy,
    shannon_entropy,
)


def test_fair_coin_is_one_bit():
    assert abs(shannon_entropy([0.5, 0.5]) - 1.0) < 1e-12


def test_certain_outcome_has_no_entropy():
    assert shannon_entropy([1.0, 0.0]) == 0.0


def test_uniform_over_eight_is_three_bits():
    assert abs(shannon_entropy([1] * 8) - 3.0) < 1e-12


def test_entropy_falls_as_the_bet_gets_lopsided():
    values = [shannon_entropy([p, 1 - p]) for p in (0.5, 0.7, 0.9, 0.99)]
    assert values == sorted(values, reverse=True)


def test_counts_are_normalized():
    assert abs(shannon_entropy([20, 20]) - shannon_entropy([0.5, 0.5])) < 1e-12


def test_natural_log_base_gives_nats():
    assert abs(shannon_entropy([0.5, 0.5], base=np.e) - np.log(2)) < 1e-12


def test_entropy_from_samples_matches_the_distribution():
    samples = ["up"] * 30 + ["down"] * 10
    assert abs(entropy_from_samples(samples) - shannon_entropy([0.75, 0.25])) < 1e-12


def test_normalized_entropy_is_one_for_uniform():
    assert abs(normalized_entropy([0.25] * 4) - 1.0) < 1e-12


def test_normalized_entropy_is_zero_for_certainty():
    assert normalized_entropy([1.0, 0.0, 0.0]) == 0.0
    assert normalized_entropy([1.0]) == 0.0


def test_kl_divergence_is_zero_for_identical_distributions():
    assert abs(kl_divergence([0.3, 0.7], [0.3, 0.7])) < 1e-12


def test_kl_divergence_is_never_negative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.random(5)
        q = rng.random(5)
        assert kl_divergence(p.tolist(), q.tolist()) >= -1e-12


def test_kl_divergence_is_not_symmetric():
    forward = kl_divergence([0.9, 0.1], [0.5, 0.5])
    backward = kl_divergence([0.5, 0.5], [0.9, 0.1])
    assert abs(forward - backward) > 1e-6


def test_kl_divergence_grows_with_the_gap():
    close = kl_divergence([0.5, 0.5], [0.55, 0.45])
    far = kl_divergence([0.5, 0.5], [0.9, 0.1])
    assert far > close


def test_kl_divergence_infinite_when_q_rules_out_a_real_outcome():
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf")


def test_kl_divergence_ignores_outcomes_p_never_visits():
    assert np.isfinite(kl_divergence([1.0, 0.0], [0.5, 0.5]))


def test_mutual_information_is_zero_for_independent_labels():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 4, 6000).tolist()
    y = rng.integers(0, 4, 6000).tolist()
    assert mutual_information(x, y) < 0.02


def test_mutual_information_equals_entropy_for_a_copy():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 4, 2000).tolist()
    assert abs(mutual_information(x, x) - entropy_from_samples(x)) < 1e-12


def test_mutual_information_is_symmetric():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 3, 1000).tolist()
    y = [(v + rng.integers(0, 2)) % 3 for v in x]
    assert abs(mutual_information(x, y) - mutual_information(y, x)) < 1e-12


def test_mutual_information_sees_a_nonlinear_link():
    rng = np.random.default_rng(4)
    values = rng.normal(0, 1, 4000)
    x = discretize(values.tolist(), bins=6)
    # Correlation with the absolute value is near zero but the link is real.
    y = discretize(np.abs(values).tolist(), bins=6)
    assert abs(np.corrcoef(values, np.abs(values))[0, 1]) < 0.1
    assert mutual_information(x, y) > 0.5


def test_discretize_splits_a_range_into_bins():
    labels = discretize([0.0, 1.0, 2.0, 3.0], bins=2)
    assert labels[:2] == [0, 0]
    assert labels[2:] == [1, 1]


def test_discretize_handles_a_flat_series():
    assert discretize([5.0, 5.0, 5.0], bins=4) == [0, 0, 0]


def test_negative_probabilities_raise():
    with pytest.raises(ValueError):
        shannon_entropy([0.5, -0.5])


def test_empty_or_zero_distribution_raises():
    with pytest.raises(ValueError):
        shannon_entropy([0.0, 0.0])
    with pytest.raises(ValueError):
        entropy_from_samples([])


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        mutual_information([1, 2, 3], [1, 2])


def test_bad_bin_count_raises():
    with pytest.raises(ValueError):
        discretize([1.0, 2.0], bins=0)
