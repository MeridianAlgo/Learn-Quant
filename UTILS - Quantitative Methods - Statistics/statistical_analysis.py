"""
Statistics Utility - Essential Statistical Analysis for Quantitative Finance

This module provides comprehensive statistical analysis tools for:
- Financial data analysis and risk management
- Portfolio performance measurement
- Hypothesis testing and statistical inference
- Monte Carlo simulations
- Distribution analysis and modeling

Author: MeridianAlgo
Version: 1.0.0
"""

from typing import Dict, Optional, Union

import numpy as np
from scipy import stats


class StatisticalAnalysis:
    """
    Comprehensive statistical analysis for quantitative finance.

    This class provides methods for:
    - Descriptive statistics and data analysis
    - Probability distributions and modeling
    - Hypothesis testing and statistical inference
    - Risk metrics and portfolio analysis
    - Monte Carlo simulations
    """

    def __init__(self):
        """Initialize the StatisticalAnalysis class."""
        self.rng = np.random.default_rng(42)

    def calculate_descriptive_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive descriptive statistics.

        Args:
            data: Input data array

        Returns:
            Dict[str, float]: Descriptive statistics
        """
        if len(data) == 0:
            return {}

        return {
            "count": len(data),
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data, ddof=1),  # Sample standard deviation
            "var": np.var(data, ddof=1),  # Sample variance
            "min": np.min(data),
            "max": np.max(data),
            "range": np.max(data) - np.min(data),
            "q25": np.percentile(data, 25),
            "q75": np.percentile(data, 75),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
            "skewness": stats.skew(data),
            "kurtosis": stats.kurtosis(data),
            "coefficient_of_variation": (np.std(data, ddof=1) / abs(np.mean(data)) if np.mean(data) != 0 else 0),
        }

    def calculate_financial_returns(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate financial return statistics.

        Args:
            prices: Array of asset prices

        Returns:
            Dict[str, float]: Financial return statistics
        """
        if len(prices) < 2:
            return {}

        returns = np.diff(prices) / prices[:-1]

        cumulative_return = (prices[-1] / prices[0]) - 1
        annualized_return = np.mean(returns) * 252  # Assuming daily returns
        annualized_volatility = np.std(returns, ddof=1) * np.sqrt(252)

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        # Risk metrics
        var_95 = np.percentile(returns, 5)  # 95% Value at Risk
        cvar_95 = self.calculate_expected_shortfall(returns, 0.95)

        return {
            "total_return": cumulative_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_return": np.max(returns),
            "min_return": np.min(returns),
            "positive_returns_ratio": np.mean(returns > 0),
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": self.calculate_max_drawdown(prices),
            "calmar_ratio": (
                annualized_return / abs(self.calculate_max_drawdown(prices))
                if self.calculate_max_drawdown(prices) != 0
                else 0
            ),
        }

    def calculate_value_at_risk(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk using different methods.

        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95)
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            float: Value at Risk
        """
        if method == "historical":
            return -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(1 - confidence_level)
            return -(mu + z_score * sigma)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")

    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            float: Expected Shortfall
        """
        var_threshold = self.calculate_value_at_risk(returns, confidence_level, "historical")
        tail_losses = returns[returns <= -var_threshold]
        return -np.mean(tail_losses) if len(tail_losses) > 0 else 0

    def calculate_max_drawdown(self, price_series: np.ndarray) -> float:
        """
        Calculate maximum drawdown from price series.

        Args:
            price_series: Array of prices

        Returns:
            float: Maximum drawdown
        """
        if len(price_series) < 2:
            return 0.0

        peak = price_series[0]
        max_drawdown = 0.0

        for price in price_series:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def normality_tests(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Perform normality tests on data.

        Args:
            data: Input data array

        Returns:
            Dict[str, Dict[str, float]]: Test results
        """
        return {
            "shapiro_wilk": {
                "statistic": stats.shapiro(data)[0],
                "p_value": stats.shapiro(data)[1],
            },
            "kolmogorov_smirnov": {
                "statistic": stats.kstest(data, "norm")[0],
                "p_value": stats.kstest(data, "norm")[1],
            },
            "anderson_darling": stats.anderson(data, dist="norm"),
            "jarque_bera": stats.jarque_bera(data),
        }

    def correlation_analysis(self, returns_matrix: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Analyze correlations between multiple assets.

        Args:
            returns_matrix: 2D array of returns (assets x time)

        Returns:
            Dict[str, Union[float, np.ndarray]]: Correlation analysis
        """
        if returns_matrix.ndim == 1:
            returns_matrix = returns_matrix.reshape(1, -1)

        correlation_matrix = np.corrcoef(returns_matrix)

        # Calculate average correlation (excluding diagonal)
        n_assets = correlation_matrix.shape[0]
        avg_correlation = (np.sum(correlation_matrix) - n_assets) / (n_assets * (n_assets - 1))

        return {
            "correlation_matrix": correlation_matrix,
            "average_correlation": avg_correlation,
            "max_correlation": np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
            "min_correlation": np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
            "diversification_ratio": (n_assets / (1 + (n_assets - 1) * avg_correlation) if avg_correlation < 1 else 1),
        }

    def hypothesis_testing(
        self, sample1: np.ndarray, sample2: np.ndarray, test_type: str = "parametric"
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform hypothesis tests between two samples.

        Args:
            sample1: First sample
            sample2: Second sample
            test_type: 'parametric' or 'nonparametric'

        Returns:
            Dict[str, Dict[str, float]]: Test results
        """
        results = {}

        if test_type == "parametric":
            # t-test for means
            t_test = stats.ttest_ind(sample1, sample2, equal_var=False)

            # F-test for variances
            f_test = stats.f_oneway(sample1, sample2)

            results = {
                "t_test": {"statistic": t_test[0], "p_value": t_test[1]},
                "f_test": {"statistic": f_test[0], "p_value": f_test[1]},
            }

        elif test_type == "nonparametric":
            # Mann-Whitney U test
            mann_whitney = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")

            # Kolmogorov-Smirnov test
            ks_test = stats.ks_2samp(sample1, sample2)

            results = {
                "mann_whitney": {
                    "statistic": mann_whitney[0],
                    "p_value": mann_whitney[1],
                },
                "ks_test": {"statistic": ks_test[0], "p_value": ks_test[1]},
            }

        return results

    def linear_regression(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform linear regression analysis.

        Args:
            x: Independent variable
            y: Dependent variable

        Returns:
            Dict[str, float]: Regression results
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_error": std_err,
            "f_statistic": ((r_value**2) / (1 - r_value**2) * (len(x) - 2) if len(x) > 2 else 0),
        }

    def monte_carlo_simulation(
        self,
        initial_value: float,
        expected_return: float,
        volatility: float,
        time_horizon: float,
        n_simulations: int = 10000,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform Monte Carlo simulation for portfolio valuation.

        Args:
            initial_value: Initial portfolio value
            expected_return: Expected annual return
            volatility: Annual volatility
            time_horizon: Time horizon in years
            n_simulations: Number of simulations

        Returns:
            Dict[str, Union[float, np.ndarray]]: Simulation results
        """
        # Generate random returns
        random_returns = self.rng.normal(expected_return, volatility, n_simulations)

        # Calculate final values
        final_values = initial_value * np.exp(random_returns * time_horizon)

        # Risk metrics
        var_95 = np.percentile(final_values, 5)
        cvar_95 = np.mean(final_values[final_values <= var_95])

        return {
            "final_values": final_values,
            "mean_final_value": np.mean(final_values),
            "median_final_value": np.median(final_values),
            "std_final_value": np.std(final_values),
            "var_95": var_95,
            "cvar_95": cvar_95,
            "probability_of_loss": np.mean(final_values < initial_value),
            "best_case": np.max(final_values),
            "worst_case": np.min(final_values),
        }

    def distribution_analysis(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Analyze various probability distributions.

        Args:
            data: Input data

        Returns:
            Dict[str, Dict[str, float]]: Distribution analysis
        """
        # Normal distribution
        mu, sigma = stats.norm.fit(data)

        # Log-normal distribution (for positive data)
        if np.all(data > 0):
            log_data = np.log(data)
            log_mu, log_sigma = stats.norm.fit(log_data)
        else:
            log_mu, log_sigma = None, None

        # Student's t-distribution
        df, t_mu, t_sigma = stats.t.fit(data)

        return {
            "normal": {
                "mean": mu,
                "std": sigma,
                "test_statistic": stats.shapiro(data)[0],
                "test_p_value": stats.shapiro(data)[1],
            },
            "lognormal": {
                "log_mean": log_mu,
                "log_std": log_sigma,
                "expected_value": (np.exp(log_mu + log_sigma**2 / 2) if log_mu is not None else None),
            },
            "t_distribution": {"degrees_of_freedom": df, "mean": t_mu, "std": t_sigma},
        }

    def rolling_statistics(self, data: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """
        Calculate rolling statistics for time series.

        Args:
            data: Time series data
            window: Rolling window size

        Returns:
            Dict[str, np.ndarray]: Rolling statistics
        """
        if len(data) < window:
            return {}

        rolling_mean = np.array([np.mean(data[i : i + window]) for i in range(len(data) - window + 1)])
        rolling_std = np.array([np.std(data[i : i + window], ddof=1) for i in range(len(data) - window + 1)])
        rolling_var = rolling_std**2

        return {
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "rolling_var": rolling_var,
            "rolling_sharpe": rolling_mean / rolling_std * np.sqrt(252),  # Annualized
        }

    def portfolio_performance_metrics(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio performance metrics.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns (optional)

        Returns:
            Dict[str, float]: Performance metrics
        """
        metrics = self.calculate_financial_returns(portfolio_returns * 100)  # Convert to percentage

        if benchmark_returns is not None:
            # Alpha and beta calculation
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns, ddof=1)

            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = metrics["annualized_return"] - (0.02 + beta * (np.mean(benchmark_returns) - 0.02))

            # Tracking error
            tracking_error = np.std(portfolio_returns - benchmark_returns, ddof=1) * np.sqrt(252)

            # Information ratio
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0

            metrics.update(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                }
            )

        return metrics

    def generate_sample_data(self, distribution: str = "normal", n_samples: int = 1000, **params) -> np.ndarray:
        """
        Generate sample data from various distributions.

        Args:
            distribution: Distribution type ('normal', 'lognormal', 't', 'uniform')
            n_samples: Number of samples
            **params: Distribution parameters

        Returns:
            np.ndarray: Generated data
        """
        if distribution == "normal":
            return self.rng.normal(params.get("mean", 0), params.get("std", 1), n_samples)
        elif distribution == "lognormal":
            return self.rng.lognormal(params.get("mean", 0), params.get("sigma", 1), n_samples)
        elif distribution == "t":
            return stats.t.rvs(params.get("df", 5), size=n_samples)
        elif distribution == "uniform":
            return self.rng.uniform(params.get("low", 0), params.get("high", 1), n_samples)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")


def main():
    """
    Main function demonstrating statistical analysis for finance.
    """
    print("=== Statistical Analysis for Quantitative Finance ===\n")

    # Initialize the class
    stats_analysis = StatisticalAnalysis()

    # Demo 1: Descriptive Statistics
    print("1. Descriptive Statistics:")
    np.random.seed(42)
    sample_data = np.random.normal(0.001, 0.02, 1000)  # Daily returns
    desc_stats = stats_analysis.calculate_descriptive_stats(sample_data)

    print(f"Mean: {desc_stats['mean']:.6f}")
    print(f"Standard Deviation: {desc_stats['std']:.6f}")
    print(f"Skewness: {desc_stats['skewness']:.6f}")
    print(f"Kurtosis: {desc_stats['kurtosis']:.6f}\n")

    # Demo 2: Financial Returns Analysis
    print("2. Financial Returns Analysis:")
    prices = np.random.lognormal(0, 0.2, 1000).cumprod() * 100  # Simulated stock prices
    return_stats = stats_analysis.calculate_financial_returns(prices)

    print(f"Total Return: {return_stats['total_return']:.4f}")
    print(f"Annualized Volatility: {return_stats['annualized_volatility']:.4f}")
    print(f"Sharpe Ratio: {return_stats['sharpe_ratio']:.4f}")
    print(f"95% VaR: {return_stats['var_95']:.6f}\n")

    # Demo 3: Risk Metrics
    print("3. Risk Metrics:")
    var_95 = stats_analysis.calculate_value_at_risk(sample_data, 0.95, "parametric")
    cvar_95 = stats_analysis.calculate_expected_shortfall(sample_data, 0.95)

    print(f"Parametric VaR (95%): {var_95:.6f}")
    print(f"Conditional VaR (95%): {cvar_95:.6f}")
    print(f"Max Drawdown: {stats_analysis.calculate_max_drawdown(prices):.4f}\n")

    # Demo 4: Monte Carlo Simulation
    print("4. Monte Carlo Simulation:")
    simulation_results = stats_analysis.monte_carlo_simulation(
        initial_value=100000,
        expected_return=0.08,
        volatility=0.15,
        time_horizon=1,
        n_simulations=5000,
    )

    print(f"Expected Final Value: ${simulation_results['mean_final_value']:,.0f}")
    print(f"95% VaR: ${simulation_results['var_95']:,.0f}")
    print(f"Probability of Loss: {simulation_results['probability_of_loss']:.2%}")

    # Demo 5: Correlation Analysis
    print("\n5. Correlation Analysis:")
    n_assets = 4
    returns_matrix = np.random.multivariate_normal(
        [0.001, 0.0008, 0.0012, 0.0009], np.eye(n_assets) * 0.0004 + 0.0001, 500
    )

    corr_analysis = stats_analysis.correlation_analysis(returns_matrix)
    print(f"Average Correlation: {corr_analysis['average_correlation']:.4f}")
    print(f"Diversification Ratio: {corr_analysis['diversification_ratio']:.4f}\n")

    # Demo 6: Distribution Analysis
    print("6. Distribution Analysis:")
    dist_analysis = stats_analysis.distribution_analysis(sample_data)
    print(f"Normal Distribution Test p-value: {dist_analysis['normal']['test_p_value']:.4f}")
    print(f"Log-Normal Expected Value: {dist_analysis['lognormal']['expected_value']:.4f}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
