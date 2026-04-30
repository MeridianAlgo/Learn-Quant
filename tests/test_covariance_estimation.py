import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Covariance Estimation"))
from covariance_estimation import (
    condition_number,
    constant_correlation_shrinkage,
    ewma_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)


@pytest.fixture
def returns_matrix():
    np.random.seed(42)
    T, N = 120, 10
    return np.random.randn(T, N) * 0.01


def test_sample_covariance_shape(returns_matrix):
    S = sample_covariance(returns_matrix)
    N = returns_matrix.shape[1]
    assert S.shape == (N, N)


def test_sample_covariance_symmetric(returns_matrix):
    S = sample_covariance(returns_matrix)
    assert np.allclose(S, S.T)


def test_lw_shrunk_shape(returns_matrix):
    result = ledoit_wolf_shrinkage(returns_matrix)
    N = returns_matrix.shape[1]
    assert result["shrunk_cov"].shape == (N, N)


def test_lw_alpha_in_range(returns_matrix):
    result = ledoit_wolf_shrinkage(returns_matrix)
    assert 0.0 <= result["alpha"] <= 1.0


def test_lw_better_conditioned_than_sample(returns_matrix):
    S = sample_covariance(returns_matrix)
    lw = ledoit_wolf_shrinkage(returns_matrix)
    assert condition_number(lw["shrunk_cov"]) <= condition_number(S) + 1  # Allow small tolerance


def test_cc_shrinkage_shape(returns_matrix):
    result = constant_correlation_shrinkage(returns_matrix)
    N = returns_matrix.shape[1]
    assert result["shrunk_cov"].shape == (N, N)


def test_cc_alpha_in_range(returns_matrix):
    result = constant_correlation_shrinkage(returns_matrix)
    assert 0.0 <= result["alpha"] <= 1.0


def test_cc_mean_correlation_in_range(returns_matrix):
    result = constant_correlation_shrinkage(returns_matrix)
    assert -1.0 <= result["mean_correlation"] <= 1.0


def test_ewma_covariance_shape(returns_matrix):
    ew = ewma_covariance(returns_matrix)
    N = returns_matrix.shape[1]
    assert ew.shape == (N, N)


def test_ewma_positive_semidefinite(returns_matrix):
    ew = ewma_covariance(returns_matrix)
    eigenvalues = np.linalg.eigvalsh(ew)
    assert np.all(eigenvalues >= -1e-10)


def test_condition_number_identity():
    cov = np.eye(5)
    assert abs(condition_number(cov) - 1.0) < 1e-6
