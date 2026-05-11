"""Linear Algebra for Quantitative Finance.

Run with:
    python linear_algebra_tutorial.py

This module teaches essential linear algebra concepts for portfolio optimization,
risk modeling, factor models, and quantitative finance.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def intro() -> None:
    """Print orientation details for the learner."""
    print("\n" + "#" * 60)
    print("QUANTITATIVE METHODS – LINEAR ALGEBRA")
    print("#" * 60)
    print("Essential linear algebra for portfolio optimization,")
    print("covariance matrices, eigenvalues, and factor models.\n")


def vectors_and_operations() -> None:
    """Demonstrate vector operations."""
    print("=" * 60)
    print("VECTORS AND OPERATIONS")
    print("=" * 60)

    # Example 1: Portfolio weights as vectors
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    print("Portfolio Weights (Vector):")
    for ticker, weight in zip(tickers, weights):
        print(f"  {ticker}: {weight:.1%}")

    # Vector properties
    print(f"\nSum of weights: {np.sum(weights):.2f}")
    print(f"Vector length (L2 norm): {np.linalg.norm(weights):.4f}")

    # Example 2: Expected returns as vector
    expected_returns = np.array([0.12, 0.15, 0.10, 0.14, 0.18])

    print("\nExpected Annual Returns:")
    for ticker, ret in zip(tickers, expected_returns):
        print(f"  {ticker}: {ret:.1%}")

    # Portfolio expected return (dot product)
    portfolio_return = np.dot(weights, expected_returns)
    print(f"\nPortfolio Expected Return: {portfolio_return:.2%}")

    # Example 3: Vector addition (combining portfolios)
    portfolio_A = np.array([100, 200, 150, 0, 0])
    portfolio_B = np.array([0, 0, 100, 250, 150])

    combined_portfolio = portfolio_A + portfolio_B

    print("\nCombining Portfolios:")
    print("Portfolio A:", portfolio_A)
    print("Portfolio B:", portfolio_B)
    print("Combined:   ", combined_portfolio)

    # Example 4: Scalar multiplication (levering portfolio)
    leverage = 1.5
    levered_weights = weights * leverage

    print(f"\nLevered Portfolio ({leverage}x):")
    for ticker, weight in zip(tickers, levered_weights):
        print(f"  {ticker}: {weight:.1%}")


def matrices_and_operations() -> None:
    """Demonstrate matrix operations."""
    print("\n" + "=" * 60)
    print("MATRICES AND OPERATIONS")
    print("=" * 60)

    # Example 1: Returns matrix (assets × time periods)
    # Rows: Assets (AAPL, GOOGL, MSFT)
    # Columns: Time periods (Month 1, 2, 3, 4)
    returns_matrix = np.array(
        [
            [0.02, -0.01, 0.03, 0.01],  # AAPL
            [0.01, 0.02, -0.01, 0.02],  # GOOGL
            [0.03, 0.00, 0.02, -0.01],  # MSFT
        ]
    )

    tickers = ["AAPL", "GOOGL", "MSFT"]

    print("Returns Matrix (Assets × Months):")
    print("           M1      M2      M3      M4")
    for ticker, returns in zip(tickers, returns_matrix):
        print(f"{ticker:>6}", end="  ")
        for ret in returns:
            print(f"{ret:>6.2%}", end="  ")
        print()

    # Matrix properties
    print(f"\nMatrix shape: {returns_matrix.shape}")
    print(f"Number of assets: {returns_matrix.shape[0]}")
    print(f"Number of periods: {returns_matrix.shape[1]}")

    # Example 2: Matrix transpose
    returns_transposed = returns_matrix.T

    print("\nTransposed Matrix (Months × Assets):")
    print("        AAPL   GOOGL    MSFT")
    for month, returns in enumerate(returns_transposed, 1):
        print(f"Month {month}", end="  ")
        for ret in returns:
            print(f"{ret:>6.2%}", end="  ")
        print()

    # Example 3: Matrix multiplication
    weights = np.array([0.4, 0.3, 0.3])

    # Portfolio returns over time
    portfolio_returns = weights @ returns_matrix  # Matrix multiplication

    print("\nPortfolio Returns Over Time:")
    print("Weights:", weights)
    print("\nMonthly Portfolio Returns:")
    for month, ret in enumerate(portfolio_returns, 1):
        print(f"  Month {month}: {ret:.2%}")


def covariance_correlation_matrices() -> None:
    """Demonstrate covariance and correlation matrices."""
    print("\n" + "=" * 60)
    print("COVARIANCE AND CORRELATION MATRICES")
    print("=" * 60)

    # Simulate daily returns for 3 assets over 252 trading days
    np.random.seed(42)
    n_assets = 3
    n_days = 252

    # Create correlated returns
    mean_returns = np.array([0.001, 0.0008, 0.0012])

    # Covariance matrix (symmetric, positive semi-definite)
    cov_matrix = np.array([[0.0004, 0.0002, 0.0001], [0.0002, 0.0003, 0.00015], [0.0001, 0.00015, 0.0005]])

    # Generate correlated returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)

    # Calculate sample covariance matrix
    sample_cov = np.cov(returns.T)

    tickers = ["AAPL", "GOOGL", "MSFT"]

    print("Sample Covariance Matrix:")
    print("          ", end="")
    for ticker in tickers:
        print(f"{ticker:>10}", end="")
    print()

    for i, ticker in enumerate(tickers):
        print(f"{ticker:>10}", end="")
        for j in range(n_assets):
            print(f"{sample_cov[i][j]:>10.6f}", end="")
        print()

    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(returns.T)

    print("\nCorrelation Matrix:")
    print("          ", end="")
    for ticker in tickers:
        print(f"{ticker:>10}", end="")
    print()

    for i, ticker in enumerate(tickers):
        print(f"{ticker:>10}", end="")
        for j in range(n_assets):
            print(f"{correlation_matrix[i][j]:>10.4f}", end="")
        print()

    # Portfolio variance calculation
    weights = np.array([0.4, 0.3, 0.3])

    # σ²_portfolio = w^T Σ w
    portfolio_variance = weights @ sample_cov @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Annualize (252 trading days)
    annual_volatility = portfolio_volatility * np.sqrt(252)

    print(f"\nPortfolio Weights: {weights}")
    print(f"Portfolio Daily Volatility: {portfolio_volatility:.4f}")
    print(f"Portfolio Annual Volatility: {annual_volatility:.2%}")


def eigenvalues_eigenvectors() -> None:
    """Demonstrate eigenvalues and eigenvectors for PCA and risk."""
    print("\n" + "=" * 60)
    print("EIGENVALUES AND EIGENVECTORS")
    print("=" * 60)

    # Correlation matrix from previous example
    correlation_matrix = np.array([[1.00, 0.65, 0.45], [0.65, 1.00, 0.50], [0.45, 0.50, 1.00]])

    tickers = ["AAPL", "GOOGL", "MSFT"]

    print("Correlation Matrix:")
    print("          ", end="")
    for ticker in tickers:
        print(f"{ticker:>10}", end="")
    print()

    for i, ticker in enumerate(tickers):
        print(f"{ticker:>10}", end="")
        for j in range(3):
            print(f"{correlation_matrix[i][j]:>10.4f}", end="")
        print()

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("\nEigenvalues (Principal Components):")
    for i, eigenvalue in enumerate(eigenvalues, 1):
        variance_explained = eigenvalue / np.sum(eigenvalues)
        print(f"  PC{i}: {eigenvalue:.4f} ({variance_explained:.1%} of variance)")

    print("\nEigenvectors (Factor Loadings):")
    print("          ", end="")
    for i in range(3):
        print(f"     PC{i + 1}", end="")
    print()

    for i, ticker in enumerate(tickers):
        print(f"{ticker:>10}", end="")
        for j in range(3):
            print(f"{eigenvectors[i][j]:>8.4f}", end="")
        print()

    # Interpretation
    print("\nInterpretation:")
    print("→ PC1 explains most variance (market factor)")
    print("→ Higher eigenvalues = more important risk factors")
    print("→ Used in PCA for dimensionality reduction")


def matrix_inverse_solving_systems() -> None:
    """Demonstrate matrix inverse and solving linear systems."""
    print("\n" + "=" * 60)
    print("MATRIX INVERSE AND SOLVING LINEAR SYSTEMS")
    print("=" * 60)

    # Example: Solving for optimal portfolio weights
    # Given target returns and covariance matrix

    # Expected returns
    expected_returns = np.array([0.10, 0.12, 0.08])

    # Covariance matrix
    cov_matrix = np.array([[0.04, 0.01, 0.005], [0.01, 0.05, 0.015], [0.005, 0.015, 0.03]])

    tickers = ["AAPL", "GOOGL", "MSFT"]

    print("Expected Returns:")
    for ticker, ret in zip(tickers, expected_returns):
        print(f"  {ticker}: {ret:.1%}")

    # Inverse of covariance matrix (precision matrix)
    try:
        cov_inv = np.linalg.inv(cov_matrix)

        print("\nInverse Covariance Matrix exists ✓")

        # Minimum variance portfolio (assuming 100% invested)
        # w = (Σ^-1 1) / (1^T Σ^-1 1)
        ones = np.ones(3)
        weights = (cov_inv @ ones) / (ones @ cov_inv @ ones)

        print("\nMinimum Variance Portfolio Weights:")
        for ticker, weight in zip(tickers, weights):
            print(f"  {ticker}: {weight:.2%}")

        print(f"\nWeights sum: {np.sum(weights):.4f}")

    except np.linalg.LinAlgError:
        print("\nMatrix is singular (not invertible)")


def matrix_determinant() -> None:
    """Demonstrate matrix determinant."""
    print("\n" + "=" * 60)
    print("MATRIX DETERMINANT")
    print("=" * 60)

    # Covariance matrix
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.05]])

    print("Covariance Matrix (2×2):")
    print(cov_matrix)

    # Calculate determinant
    det = np.linalg.det(cov_matrix)

    print(f"\nDeterminant: {det:.6f}")

    if det > 0:
        print("→ Matrix is positive definite (invertible) ✓")
    elif det == 0:
        print("→ Matrix is singular (not invertible)")
    else:
        print("→ Matrix is not positive semi-definite")

    # Determinant interpretation
    print("\nInterpretation:")
    print("→ Measures 'volume' in factor space")
    print("→ Zero determinant = linear dependence")
    print("→ Used to check matrix invertibility")


def practical_example_portfolio_optimization() -> None:
    """Practical example: Portfolio optimization using linear algebra."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLE: PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # Portfolio data
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    expected_returns = np.array([0.12, 0.15, 0.10, 0.14])

    # Covariance matrix (annualized)
    cov_matrix = np.array(
        [
            [0.040, 0.015, 0.018, 0.020],
            [0.015, 0.050, 0.012, 0.025],
            [0.018, 0.012, 0.035, 0.015],
            [0.020, 0.025, 0.015, 0.055],
        ]
    )

    # Equal weight portfolio
    n_assets = len(tickers)
    weights_equal = np.ones(n_assets) / n_assets

    # Calculate portfolio metrics using linear algebra
    portfolio_return = weights_equal @ expected_returns
    portfolio_variance = weights_equal @ cov_matrix @ weights_equal
    portfolio_vol = np.sqrt(portfolio_variance)
    sharpe_ratio = portfolio_return / portfolio_vol

    print("Equal-Weight Portfolio Analysis:")
    print("\nWeights:")
    for ticker, weight in zip(tickers, weights_equal):
        print(f"  {ticker}: {weight:.2%}")

    print(f"\nExpected Return: {portfolio_return:.2%}")
    print(f"Portfolio Volatility: {portfolio_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

    # Calculate contribution to risk
    marginal_risk = cov_matrix @ weights_equal
    risk_contribution = weights_equal * marginal_risk
    risk_contribution_pct = risk_contribution / portfolio_variance

    print("\nRisk Contribution:")
    for ticker, contrib in zip(tickers, risk_contribution_pct):
        print(f"  {ticker}: {contrib:.2%}")


def main() -> None:
    """Run all linear algebra examples."""
    intro()
    vectors_and_operations()
    matrices_and_operations()
    covariance_correlation_matrices()
    eigenvalues_eigenvectors()
    matrix_inverse_solving_systems()
    matrix_determinant()
    practical_example_portfolio_optimization()
    print("\n🎉 Linear Algebra tutorial complete!")
    print("Master these concepts for portfolio optimization and risk analysis.")


if __name__ == "__main__":
    main()
