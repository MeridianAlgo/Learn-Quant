import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Kernel Density Estimation"))
from kde import (
    epanechnikov_kernel,
    gaussian_kernel,
    kde,
    kde_cdf,
    kde_quantile,
    kde_var,
    sample_kde,
    scott_bandwidth,
    silverman_bandwidth,
)


@pytest.fixture
def normal_sample():
    return np.random.default_rng(0).normal(0.0, 1.0, 2000)


def test_gaussian_kernel_peak():
    assert gaussian_kernel(0.0) == pytest.approx(1 / np.sqrt(2 * np.pi))


def test_epanechnikov_support():
    assert epanechnikov_kernel(0.0) == pytest.approx(0.75)
    assert epanechnikov_kernel(1.5) == 0.0


def test_kernels_integrate_to_one():
    u = np.linspace(-6, 6, 20001)
    du = u[1] - u[0]
    assert np.sum(gaussian_kernel(u)) * du == pytest.approx(1.0, abs=1e-4)
    assert np.sum(epanechnikov_kernel(u)) * du == pytest.approx(1.0, abs=1e-3)


def test_bandwidths_positive_and_ordered(normal_sample):
    s = silverman_bandwidth(normal_sample)
    sc = scott_bandwidth(normal_sample)
    assert 0 < s < sc


def test_silverman_resists_an_outlier(normal_sample):
    spiked = np.append(normal_sample, 500.0)
    assert silverman_bandwidth(spiked) < 1.2 * silverman_bandwidth(normal_sample)
    assert scott_bandwidth(spiked) > 3 * scott_bandwidth(normal_sample)


def test_bandwidth_rejects_constant_data():
    with pytest.raises(ValueError):
        silverman_bandwidth([1.0, 1.0, 1.0])


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov"])
def test_density_integrates_to_one(normal_sample, kernel):
    grid = np.linspace(-7, 7, 4001)
    dens = kde(normal_sample, grid, kernel=kernel)
    assert np.all(dens >= 0)
    assert np.sum(dens) * (grid[1] - grid[0]) == pytest.approx(1.0, abs=1e-3)


def test_density_close_to_true_normal(normal_sample):
    est = kde(normal_sample, [0.0])[0]
    assert est == pytest.approx(1 / np.sqrt(2 * np.pi), rel=0.08)


def test_kde_validation():
    with pytest.raises(ValueError):
        kde([1.0], [0.0])
    with pytest.raises(ValueError):
        kde([1.0, 2.0], [0.0], bandwidth=0.0)
    with pytest.raises(ValueError):
        kde([1.0, 2.0], [0.0], kernel="box")
    with pytest.raises(ValueError):
        kde([1.0, np.nan], [0.0])


def test_cdf_monotone_and_bounded(normal_sample):
    grid = np.linspace(-5, 5, 50)
    cdf = kde_cdf(normal_sample, grid)
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[0] < 0.001 and cdf[-1] > 0.999


def test_cdf_median_near_half(normal_sample):
    assert kde_cdf(normal_sample, 0.0)[0] == pytest.approx(0.5, abs=0.03)


def test_quantile_inverts_cdf(normal_sample):
    q = kde_quantile(normal_sample, 0.1)
    assert kde_cdf(normal_sample, q)[0] == pytest.approx(0.1, abs=1e-6)


def test_quantile_validation():
    with pytest.raises(ValueError):
        kde_quantile([0.0, 1.0], 1.0)


def test_var_is_positive_and_grows_with_confidence(normal_sample):
    rets = 0.01 * normal_sample
    v95 = kde_var(rets, 0.95)
    v99 = kde_var(rets, 0.99)
    assert 0 < v95 < v99
    assert v95 == pytest.approx(0.01 * 1.645, rel=0.1)


def test_var_validation():
    with pytest.raises(ValueError):
        kde_var([0.0, 1.0], 0.4)


def test_sampling_reproducible_and_centered(normal_sample):
    a = sample_kde(normal_sample, 5000, seed=4)
    b = sample_kde(normal_sample, 5000, seed=4)
    assert np.array_equal(a, b)
    assert abs(a.mean()) < 0.06
    # Smoothing adds the bandwidth's variance on top of the sample variance.
    assert a.var() > normal_sample.var() * 0.95


def test_sampling_validation(normal_sample):
    with pytest.raises(ValueError):
        sample_kde(normal_sample, 0)
