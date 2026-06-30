import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Hypothesis Testing"))
from hypothesis_testing import (
    confidence_interval,
    one_sample_ttest,
    reject_null,
    two_sample_ttest,
    z_test,
)


def test_one_sample_zero_mean_is_not_significant():
    # Symmetric data centred on zero should not reject the null.
    sample = [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
    res = one_sample_ttest(sample, mu0=0.0)
    assert abs(res["t"]) < 1e-9
    assert res["p_value"] > 0.99


def test_one_sample_clear_positive_mean_is_significant():
    sample = [5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.95]
    res = one_sample_ttest(sample, mu0=0.0)
    assert res["t"] > 0
    assert res["p_value"] < 0.001
    assert reject_null(res["p_value"]) is True


def test_one_sample_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    sample = [0.12, -0.05, 0.20, 0.08, -0.10, 0.15, 0.03, 0.18]
    res = one_sample_ttest(sample, mu0=0.0)
    t_ref, p_ref = scipy_stats.ttest_1samp(sample, 0.0)
    assert res["t"] == pytest.approx(t_ref, rel=1e-9)
    assert res["p_value"] == pytest.approx(p_ref, rel=1e-9)


def test_welch_two_sample_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    a = [0.10, 0.12, 0.09, 0.14, 0.11, 0.13, 0.08]
    b = [0.06, 0.05, 0.08, 0.04, 0.07, 0.03, 0.06]
    res = two_sample_ttest(a, b, equal_var=False)
    t_ref, p_ref = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert res["t"] == pytest.approx(t_ref, rel=1e-9)
    assert res["p_value"] == pytest.approx(p_ref, rel=1e-9)


def test_pooled_two_sample_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 3.0, 4.0, 5.0, 6.0]
    res = two_sample_ttest(a, b, equal_var=True)
    t_ref, p_ref = scipy_stats.ttest_ind(a, b, equal_var=True)
    assert res["t"] == pytest.approx(t_ref, rel=1e-9)
    assert res["p_value"] == pytest.approx(p_ref, rel=1e-9)


def test_z_test_zero_when_mean_equals_mu0():
    res = z_test([3.0, 3.0, 3.0], mu0=3.0, sigma=1.0)
    assert res["z"] == pytest.approx(0.0)
    assert res["p_value"] == pytest.approx(1.0)


def test_confidence_interval_brackets_the_mean():
    sample = [10, 11, 9, 12, 8, 10, 11, 9]
    low, high = confidence_interval(sample, 0.95)
    mean = sum(sample) / len(sample)
    assert low < mean < high
    # A wider confidence level gives a wider interval.
    low99, high99 = confidence_interval(sample, 0.99)
    assert (high99 - low99) > (high - low)


def test_reject_null_threshold():
    assert reject_null(0.01, alpha=0.05) is True
    assert reject_null(0.10, alpha=0.05) is False


def test_errors_on_degenerate_input():
    with pytest.raises(ValueError):
        one_sample_ttest([1.0], mu0=0.0)
    with pytest.raises(ValueError):
        one_sample_ttest([5.0, 5.0, 5.0], mu0=0.0)  # zero variance
    with pytest.raises(ValueError):
        z_test([1.0, 2.0], mu0=0.0, sigma=-1.0)
