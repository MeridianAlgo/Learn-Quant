"""
Test suite for Quantitative Methods - Linear Algebra.

Run with:
    python -m pytest tests/test_linear_algebra.py -v

Or directly:
    python tests/test_linear_algebra.py
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vector_operations():
    """Test basic vector operations."""
    # Portfolio weights
    weights = np.array([0.3, 0.3, 0.4])

    assert np.isclose(np.sum(weights), 1.0), "Weights should sum to 1"

    # Expected returns
    expected_returns = np.array([0.10, 0.12, 0.08])

    # Portfolio return (dot product)
    portfolio_return = np.dot(weights, expected_returns)
    expected_value = 0.3 * 0.10 + 0.3 * 0.12 + 0.4 * 0.08

    assert np.isclose(
        portfolio_return, expected_value
    ), "Portfolio return calculation incorrect"

    print("✓ Vector operations tests passed")


def test_matrix_multiplication():
    """Test matrix multiplication for portfolio returns."""
    # Returns matrix: 3 assets × 4 time periods
    returns = np.array(
        [
            [0.02, -0.01, 0.03, 0.01],
            [0.01, 0.02, -0.01, 0.02],
            [0.03, 0.00, 0.02, -0.01],
        ]
    )

    weights = np.array([0.4, 0.3, 0.3])

    # Portfolio returns over time
    portfolio_returns = weights @ returns

    assert portfolio_returns.shape == (4,), "Portfolio returns shape incorrect"

    # Check first period manually
    expected_first = 0.4 * 0.02 + 0.3 * 0.01 + 0.3 * 0.03
    assert np.isclose(
        portfolio_returns[0], expected_first
    ), "First period return incorrect"

    print("✓ Matrix multiplication tests passed")


def test_covariance_matrix():
    """Test covariance matrix calculations."""
    # Simple returns data
    returns = np.array([[0.01, 0.02, 0.015], [0.02, 0.01, 0.025], [0.015, 0.025, 0.02]])

    # Calculate covariance matrix
    cov_matrix = np.cov(returns)

    # Covariance matrix should be symmetric
    assert np.allclose(cov_matrix, cov_matrix.T), "Covariance matrix not symmetric"

    # Diagonal elements should be variances (positive)
    assert np.all(np.diag(cov_matrix) >= 0), "Variances must be non-negative"

    print("✓ Covariance matrix tests passed")


def test_portfolio_variance():
    """Test portfolio variance calculation."""
    weights = np.array([0.5, 0.5])

    # Covariance matrix for 2 assets
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.05]])

    # σ²_portfolio = w^T Σ w
    portfolio_variance = weights @ cov_matrix @ weights

    # Manual calculation
    expected_variance = (
        0.5 * 0.5 * 0.04  # w1² σ1²
        + 0.5 * 0.5 * 0.05  # w2² σ2²
        + 2 * 0.5 * 0.5 * 0.01  # 2 w1 w2 σ12
    )

    assert np.isclose(
        portfolio_variance, expected_variance
    ), "Portfolio variance incorrect"

    # Variance should be positive
    assert portfolio_variance > 0, "Variance must be positive"

    print("✓ Portfolio variance tests passed")


def test_correlation_matrix():
    """Test correlation matrix calculations."""
    # Generate correlated returns
    np.random.seed(42)
    returns = np.random.multivariate_normal(
        mean=[0.001, 0.0008], cov=[[0.0004, 0.0002], [0.0002, 0.0003]], size=100
    )

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(returns.T)

    # Diagonal should be 1
    assert np.allclose(
        np.diag(corr_matrix), 1.0
    ), "Diagonal of correlation matrix must be 1"

    # Symmetric
    assert np.allclose(corr_matrix, corr_matrix.T), "Correlation matrix not symmetric"

    # Values between -1 and 1
    assert np.all(corr_matrix >= -1) and np.all(
        corr_matrix <= 1
    ), "Correlation values out of range"

    print("✓ Correlation matrix tests passed")


def test_eigenvalues_eigenvectors():
    """Test eigenvalue decomposition."""
    # Symmetric positive semi-definite matrix
    matrix = np.array([[2, 1], [1, 2]])

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Check eigenvalue equation: A v = λ v
    for i in range(len(eigenvalues)):
        lhs = matrix @ eigenvectors[:, i]
        rhs = eigenvalues[i] * eigenvectors[:, i]
        assert np.allclose(lhs, rhs), f"Eigenvalue equation failed for eigenvalue {i}"

    # For symmetric matrix, eigenvalues should be real
    assert np.all(
        np.isreal(eigenvalues)
    ), "Eigenvalues of symmetric matrix must be real"

    print("✓ Eigenvalue/eigenvector tests passed")


def test_matrix_inverse():
    """Test matrix inversion."""
    # Invertible matrix
    matrix = np.array([[4, 7], [2, 6]])

    # Calculate inverse
    matrix_inv = np.linalg.inv(matrix)

    # Check A A^-1 = I
    identity = matrix @ matrix_inv
    expected_identity = np.eye(2)

    assert np.allclose(identity, expected_identity), "Matrix inverse incorrect"

    print("✓ Matrix inverse tests passed")


def test_determinant():
    """Test matrix determinant."""
    # 2x2 matrix
    matrix = np.array([[4, 7], [2, 6]])

    det = np.linalg.det(matrix)

    # Manual calculation for 2x2: ad - bc
    expected_det = 4 * 6 - 7 * 2

    assert np.isclose(det, expected_det), "Determinant calculation incorrect"

    # Singular matrix (determinant = 0)
    singular_matrix = np.array([[1, 2], [2, 4]])

    singular_det = np.linalg.det(singular_matrix)
    assert np.isclose(singular_det, 0), "Singular matrix should have determinant 0"

    print("✓ Determinant tests passed")


def test_minimum_variance_portfolio():
    """Test minimum variance portfolio calculation."""
    # Covariance matrix
    cov_matrix = np.array(
        [[0.04, 0.01, 0.008], [0.01, 0.05, 0.012], [0.008, 0.012, 0.03]]
    )

    # Minimum variance portfolio: w = (Σ^-1 1) / (1^T Σ^-1 1)
    cov_inv = np.linalg.inv(cov_matrix)
    ones = np.ones(3)

    weights = (cov_inv @ ones) / (ones @ cov_inv @ ones)

    # Weights should sum to 1
    assert np.isclose(np.sum(weights), 1.0), "Weights must sum to 1"

    # All weights should be positive for long-only portfolio
    # (This might not always be true depending on the covariance matrix)

    # Calculate portfolio variance
    portfolio_var = weights @ cov_matrix @ weights

    # This should be the minimum possible variance
    # Test with random weights
    np.random.seed(42)
    random_weights = np.random.random(3)
    random_weights /= np.sum(random_weights)
    random_var = random_weights @ cov_matrix @ random_weights

    assert (
        portfolio_var <= random_var + 1e-10
    ), "Minimum variance portfolio should have lower variance"

    print("✓ Minimum variance portfolio tests passed")


def test_transpose():
    """Test matrix transpose."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    transposed = matrix.T

    # Check dimensions
    assert transposed.shape == (3, 2), "Transpose shape incorrect"

    # Check values
    assert transposed[0, 0] == 1 and transposed[0, 1] == 4, "Transpose values incorrect"
    assert transposed[1, 0] == 2 and transposed[1, 1] == 5, "Transpose values incorrect"

    print("✓ Transpose tests passed")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("Testing Quantitative Methods - Linear Algebra")
    print("=" * 60 + "\n")

    test_vector_operations()
    test_matrix_multiplication()
    test_covariance_matrix()
    test_portfolio_variance()
    test_correlation_matrix()
    test_eigenvalues_eigenvectors()
    test_matrix_inverse()
    test_determinant()
    test_minimum_variance_portfolio()
    test_transpose()

    print("\n" + "=" * 60)
    print("All Linear Algebra tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
