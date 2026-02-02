"""
Arrays Utility - NumPy Fundamentals for Quantitative Finance

This module provides comprehensive NumPy array operations essential for:
- Financial time series analysis
- Portfolio optimization
- Risk modeling
- Monte Carlo simulations
- Linear algebra applications

Author: MeridianAlgo
Version: 1.0.0
"""

from typing import Tuple

import numpy as np
from scipy import stats


class ArrayOperations:
    """
    Comprehensive array operations for quantitative finance applications.

    This class provides methods for:
    - Array creation and manipulation
    - Financial calculations
    - Risk analysis
    - Portfolio optimization
    - Monte Carlo simulations
    """

    def __init__(self):
        """Initialize the ArrayOperations class."""
        self.rng = np.random.default_rng(42)  # For reproducible results

    def beginner_array_basics(self) -> None:
        """Provide a beginner-friendly walkthrough of NumPy arrays."""
        print("=== Beginner Array Basics ===")

        # 1. Create arrays from Python lists
        python_list = [1, 2, 3, 4, 5]
        numpy_array = np.array(python_list)
        print(f"Original Python list: {python_list}")
        print(f"Converted NumPy array: {numpy_array}")

        # 2. Show array properties
        print(f"Array shape: {numpy_array.shape}")
        print(f"Array data type: {numpy_array.dtype}")

        # 3. Demonstrate vectorized operations
        doubled = numpy_array * 2
        print(f"Doubled values (vectorized): {doubled}")

        # 4. Explain basic statistics
        print(f"Mean of the array: {numpy_array.mean():.2f}")
        print(f"Standard deviation: {numpy_array.std():.2f}")

        # 5. Show slicing
        print(f"First three elements (slicing): {numpy_array[:3]}")

        print(
            "Python Tip: Arrays start at index 0, so [:3] grabs positions 0, 1, and 2.\n"
        )

    def create_price_series(
        self,
        start_price: float,
        periods: int,
        drift: float = 0.0,
        volatility: float = 0.1,
    ) -> np.ndarray:
        """
        Create a simulated price series using geometric Brownian motion.

        Args:
            start_price: Initial price
            periods: Number of time periods
            drift: Expected annual return (default: 0.0)
            volatility: Annual volatility (default: 0.1)

        Returns:
            np.ndarray: Array of price values
        """
        dt = 1 / 252  # Daily time step (assuming 252 trading days)

        # Generate random shocks
        shocks = self.rng.standard_normal(periods) * np.sqrt(dt)

        # Calculate price path
        price_series = np.zeros(periods + 1)
        price_series[0] = start_price

        for i in range(1, periods + 1):
            price_series[i] = price_series[i - 1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * shocks[i - 1]
            )

        return price_series

    def calculate_returns(
        self, prices: np.ndarray, method: str = "arithmetic"
    ) -> np.ndarray:
        """
        Calculate returns from price series.

        Args:
            prices: Array of price values
            method: 'arithmetic' or 'logarithmic'

        Returns:
            np.ndarray: Array of return values
        """
        if method == "arithmetic":
            returns = np.diff(prices) / prices[:-1]
        elif method == "logarithmic":
            returns = np.diff(np.log(prices))
        else:
            raise ValueError("Method must be 'arithmetic' or 'logarithmic'")

        return returns

    def portfolio_return(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate portfolio return given weights and asset returns.

        Args:
            weights: Portfolio weights (must sum to 1)
            returns: Asset returns array

        Returns:
            float: Portfolio return
        """
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        return np.dot(weights, returns)

    def portfolio_volatility(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio volatility.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix of asset returns

        Returns:
            float: Portfolio volatility (standard deviation)
        """
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def covariance_matrix(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate covariance matrix from returns data.

        Args:
            returns: 2D array of asset returns (assets x time periods)

        Returns:
            np.ndarray: Covariance matrix
        """
        if returns.ndim == 1:
            returns = returns.reshape(1, -1)

        return np.cov(returns)

    def correlation_matrix(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix from returns data.

        Args:
            returns: 2D array of asset returns

        Returns:
            np.ndarray: Correlation matrix
        """
        return np.corrcoef(returns)

    def optimal_portfolio_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> np.ndarray:
        """
        Calculate optimal portfolio weights using mean-variance optimization.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate for Sharpe ratio

        Returns:
            np.ndarray: Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        ones = np.ones(n_assets)

        # Set up matrices for quadratic optimization
        A = np.vstack([np.zeros((1, n_assets)), ones])
        A = np.vstack([A, np.column_stack([ones, cov_matrix])])

        # Target vector for Sharpe ratio maximization
        b = np.zeros(n_assets + 2)
        b[0] = 1  # Maximize Sharpe ratio
        b[1] = 1  # Budget constraint

        # Solve system
        solution = np.linalg.solve(A, b)
        weights = solution[2:]

        return weights / np.sum(weights)  # Normalize

    def monte_carlo_simulation(
        self, s0: float, mu: float, sigma: float, t: float, n_paths: int, n_steps: int
    ) -> np.ndarray:
        """
        Perform Monte Carlo simulation for asset price paths.

        Args:
            s0: Initial price
            mu: Expected return
            sigma: Volatility
            t: Time horizon
            n_paths: Number of simulation paths
            n_steps: Number of time steps

        Returns:
            np.ndarray: Array of simulated price paths
        """
        dt = t / n_steps
        price_paths = np.zeros((n_paths, n_steps + 1))
        price_paths[:, 0] = s0

        for i in range(1, n_steps + 1):
            z = self.rng.standard_normal(n_paths)
            price_paths[:, i] = price_paths[:, i - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )

        return price_paths

    def value_at_risk(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            float: Value at Risk
        """
        return -np.percentile(returns, (1 - confidence_level) * 100)

    def expected_shortfall(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR) using historical simulation.

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level

        Returns:
            float: Expected Shortfall
        """
        var_threshold = self.value_at_risk(returns, confidence_level)
        tail_losses = returns[returns <= -var_threshold]
        return -np.mean(tail_losses)

    def linear_regression(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Perform linear regression analysis.

        Args:
            x: Independent variable array
            y: Dependent variable array

        Returns:
            Tuple[float, float, float]: slope, intercept, r_squared
        """
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - np.sum(x) ** 2
        )

        intercept = (np.sum(y) - slope * np.sum(x)) / n

        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return slope, intercept, r_squared

    def matrix_operations_demo(self) -> None:
        """
        Demonstrate various matrix operations useful in finance.
        """
        print("=== Matrix Operations Demo ===")

        # Create sample covariance matrix
        cov_matrix = np.array(
            [[0.04, 0.02, 0.01], [0.02, 0.03, 0.015], [0.01, 0.015, 0.025]]
        )

        print("Covariance Matrix:")
        print(cov_matrix)

        # Matrix decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        print(f"\nEigenvalues: {eigenvalues}")
        print(f"Eigenvectors:\n{eigenvectors}")

        # Matrix inverse
        inv_cov = np.linalg.inv(cov_matrix)
        print(f"\nInverse Covariance Matrix:\n{inv_cov}")

        # Cholesky decomposition (for Monte Carlo)
        L = np.linalg.cholesky(cov_matrix)
        print(f"\nCholesky Decomposition (L):\n{L}")

        # Verify: L @ L.T should equal original matrix
        reconstructed = L @ L.T
        print(f"\nReconstructed Matrix (L @ L.T):\n{reconstructed}")
        print(f"Reconstruction error: {np.max(np.abs(cov_matrix - reconstructed))}")

    def statistical_analysis(self, data: np.ndarray) -> dict:
        """
        Perform comprehensive statistical analysis on financial data.

        Args:
            data: Array of financial data (returns, prices, etc.)

        Returns:
            dict: Dictionary containing statistical measures
        """
        return {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "var": np.var(data),
            "skewness": stats.skew(data),
            "kurtosis": stats.kurtosis(data),
            "min": np.min(data),
            "max": np.max(data),
            "range": np.max(data) - np.min(data),
            "q25": np.percentile(data, 25),
            "q75": np.percentile(data, 75),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
        }

    def hypothesis_testing(
        self, sample1: np.ndarray, sample2: np.ndarray, test_type: str = "t_test"
    ) -> dict:
        """
        Perform hypothesis testing between two samples.

        Args:
            sample1: First sample array
            sample2: Second sample array
            test_type: Type of test ('t_test', 'f_test', 'ks_test')

        Returns:
            dict: Test results
        """
        if test_type == "t_test":
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            return {"t_statistic": t_stat, "p_value": p_value}
        elif test_type == "f_test":
            f_stat, p_value = stats.f_oneway(sample1, sample2)
            return {"f_statistic": f_stat, "p_value": p_value}
        elif test_type == "ks_test":
            ks_stat, p_value = stats.ks_2samp(sample1, sample2)
            return {"ks_statistic": ks_stat, "p_value": p_value}
        else:
            raise ValueError("Unsupported test type")


def main():
    """
    Main function demonstrating array operations for quantitative finance.
    """
    print("=== NumPy Arrays for Quantitative Finance ===\n")

    # Initialize the class
    arrays = ArrayOperations()

    # Beginner walkthrough for newcomers
    arrays.beginner_array_basics()

    # Demo 1: Price Series Simulation
    print("1. Price Series Simulation:")
    prices = arrays.create_price_series(100, 100, 0.08, 0.20)
    returns = arrays.calculate_returns(prices, "logarithmic")
    print(f"Initial price: ${prices[0]:.2f}")
    print(f"Final price: ${prices[-1]:.2f}")
    print(f"Total return: {(prices[-1] / prices[0] - 1):.2%}")
    print(f"Volatility: {np.std(returns):.4f}\n")

    # Demo 2: Portfolio Analysis
    print("2. Portfolio Analysis:")
    n_assets = 3
    sample_returns = np.random.multivariate_normal(
        [0.001, 0.0008, 0.0012],
        arrays.covariance_matrix(np.random.randn(100, n_assets)),
        100,
    )

    weights = np.array([0.5, 0.3, 0.2])
    port_return = arrays.portfolio_return(weights, np.mean(sample_returns, axis=0))
    port_vol = arrays.portfolio_volatility(
        weights, arrays.covariance_matrix(sample_returns)
    )

    print(f"Portfolio return: {port_return:.4f}")
    print(f"Portfolio volatility: {port_vol:.4f}")
    print(f"Sharpe ratio: {port_return / port_vol:.4f}\n")

    # Demo 3: Risk Metrics
    print("3. Risk Metrics:")
    var_95 = arrays.value_at_risk(returns, 0.95)
    es_95 = arrays.expected_shortfall(returns, 0.95)
    print(f"95% VaR: {var_95:.4f}")
    print(f"95% Expected Shortfall: {es_95:.4f}\n")

    # Demo 4: Matrix Operations
    arrays.matrix_operations_demo()

    # Demo 5: Statistical Analysis
    print("\n4. Statistical Analysis:")
    stats_summary = arrays.statistical_analysis(returns)
    for key, value in stats_summary.items():
        print(f"{key}: {value:.6f}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
