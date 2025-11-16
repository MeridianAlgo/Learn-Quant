# Statistics - Essential Statistical Analysis for Quantitative Finance

## 📋 Overview

This utility provides comprehensive statistical analysis tools essential for quantitative finance, risk management, and investment analysis. Statistics forms the foundation for understanding financial data patterns, risk assessment, and predictive modeling.

## 🎯 Key Concepts

### **Descriptive Statistics**
- **Measures of Central Tendency**: Mean, median, mode
- **Measures of Dispersion**: Variance, standard deviation, range
- **Distribution Shape**: Skewness, kurtosis
- **Position Measures**: Percentiles, quartiles, z-scores

### **Probability Distributions**
- **Normal Distribution**: Stock returns, central limit theorem
- **Log-Normal Distribution**: Asset prices, positive values
- **Student's t-Distribution**: Small sample statistics
- **Chi-Square Distribution**: Variance testing, goodness of fit

### **Statistical Inference**
- **Hypothesis Testing**: t-tests, ANOVA, chi-square tests
- **Confidence Intervals**: Parameter estimation
- **Correlation Analysis**: Linear relationships
- **Regression Analysis**: Predictive modeling

### **Financial Applications**
- **Risk Metrics**: Value at Risk (VaR), Expected Shortfall
- **Portfolio Analysis**: Sharpe ratio, diversification benefits
- **Performance Measurement**: Alpha, beta, tracking error
- **Market Analysis**: Trend analysis, momentum strategies

## 💻 Implementation

### **Descriptive Statistics**
```python
import numpy as np
from scipy import stats

def calculate_descriptive_stats(data: np.ndarray) -> dict:
    """Calculate comprehensive descriptive statistics."""
    return {
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data).mode[0] if len(data) > 0 else None,
        'std': np.std(data),
        'var': np.var(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }

def calculate_financial_returns(prices: np.ndarray) -> dict:
    """Calculate financial return statistics."""
    returns = np.diff(prices) / prices[:-1]

    return {
        'total_return': (prices[-1] / prices[0]) - 1,
        'annualized_return': np.mean(returns) * 252,  # Assuming daily returns
        'annualized_volatility': np.std(returns) * np.sqrt(252),
        'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(prices),
        'var_95': np.percentile(returns, 5),  # 95% Value at Risk
        'cvar_95': calculate_conditional_var(returns, 0.95)
    }
```

### **Probability Distributions**
```python
def normal_distribution_analysis(data: np.ndarray) -> dict:
    """Analyze normal distribution properties."""
    mu, sigma = stats.norm.fit(data)

    return {
        'mean': mu,
        'std': sigma,
        'shapiro_test': stats.shapiro(data),  # Normality test
        'ks_test': stats.kstest(data, 'norm'),  # Kolmogorov-Smirnov test
        'confidence_interval_95': stats.norm.interval(0.95, mu, sigma),
        'percentile_5': stats.norm.ppf(0.05, mu, sigma),
        'percentile_95': stats.norm.ppf(0.95, mu, sigma)
    }

def lognormal_analysis(positive_data: np.ndarray) -> dict:
    """Analyze log-normal distribution for asset prices."""
    log_data = np.log(positive_data)
    mu, sigma = stats.norm.fit(log_data)

    return {
        'log_mean': mu,
        'log_std': sigma,
        'expected_value': np.exp(mu + sigma**2 / 2),
        'median': np.exp(mu),
        'mode': np.exp(mu - sigma**2),
        'variance': (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    }
```

### **Hypothesis Testing**
```python
def perform_hypothesis_tests(sample1: np.ndarray, sample2: np.ndarray) -> dict:
    """Perform comprehensive hypothesis tests."""
    return {
        't_test': stats.ttest_ind(sample1, sample2),
        'mann_whitney': stats.mannwhitneyu(sample1, sample2, alternative='two-sided'),
        'ks_test': stats.ks_2samp(sample1, sample2),
        'levene_test': stats.levene(sample1, sample2),  # Equal variances
        'f_test': stats.f_oneway(sample1, sample2)
    }

def correlation_analysis(returns_matrix: np.ndarray) -> dict:
    """Analyze correlations between assets."""
    correlations = np.corrcoef(returns_matrix)

    return {
        'correlation_matrix': correlations,
        'average_correlation': np.mean(correlations[np.triu_indices_from(correlations, k=1)]),
        'max_correlation': np.max(correlations),
        'min_correlation': np.min(correlations),
        'diversification_ratio': len(returns_matrix) / (1 + np.sum(correlations)) if len(returns_matrix) > 1 else 1
    }
```

### **Risk Metrics**
```python
def calculate_value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk using historical simulation."""
    return -np.percentile(returns, (1 - confidence_level) * 100)

def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate Expected Shortfall (Conditional VaR)."""
    var_threshold = calculate_value_at_risk(returns, confidence_level)
    tail_losses = returns[returns <= -var_threshold]
    return -np.mean(tail_losses)

def calculate_max_drawdown(price_series: np.ndarray) -> float:
    """Calculate maximum drawdown from price series."""
    peak = price_series[0]
    max_drawdown = 0

    for price in price_series:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown
```

## 📊 Examples

### **Example 1: Portfolio Risk Analysis**
```python
def portfolio_risk_analysis():
    """Comprehensive portfolio risk analysis."""
    print("=== Portfolio Risk Analysis ===")

    # Simulate portfolio returns
    np.random.seed(42)
    n_assets = 5
    n_periods = 1000

    # Generate correlated returns
    mean_returns = np.array([0.001, 0.0008, 0.0012, 0.0009, 0.0011])
    cov_matrix = np.array([
        [0.0004, 0.0002, 0.0001, 0.00015, 0.0001],
        [0.0002, 0.0003, 0.00015, 0.0001, 0.00012],
        [0.0001, 0.00015, 0.0005, 0.0002, 0.00018],
        [0.00015, 0.0001, 0.0002, 0.0004, 0.00016],
        [0.0001, 0.00012, 0.00018, 0.00016, 0.00035]
    ])

    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)

    # Portfolio weights
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

    # Calculate portfolio statistics
    portfolio_returns = np.dot(returns, weights)

    stats = calculate_descriptive_stats(portfolio_returns)
    print(f"Portfolio Mean Return: {stats['mean']".6f"}")
    print(f"Portfolio Volatility: {stats['std']".6f"}")
    print(f"Sharpe Ratio: {stats['mean'] / stats['std']".4f"}")

    # Risk metrics
    var_95 = calculate_value_at_risk(portfolio_returns, 0.95)
    cvar_95 = calculate_expected_shortfall(portfolio_returns, 0.95)
    print(f"95% VaR: {var_95".6f"}")
    print(f"95% CVaR: {cvar_95".6f"}")

    # Correlation analysis
    corr_analysis = correlation_analysis(returns)
    print(f"Average Correlation: {corr_analysis['average_correlation']".4f"}")
    print(f"Diversification Ratio: {corr_analysis['diversification_ratio']".4f"}")

    print()
```

### **Example 2: Hypothesis Testing for Market Efficiency**
```python
def market_efficiency_tests():
    """Test market efficiency using statistical methods."""
    print("=== Market Efficiency Tests ===")

    # Simulate market returns
    np.random.seed(42)
    n_periods = 500

    # Generate random walk returns (efficient market hypothesis)
    efficient_returns = np.random.normal(0.0005, 0.02, n_periods)

    # Generate momentum returns (inefficient market)
    momentum_returns = []
    momentum_returns.append(np.random.normal(0.0005, 0.02))
    for i in range(1, n_periods):
        # Add momentum component
        momentum = 0.1 * (momentum_returns[i-1] - 0.0005)
        momentum_returns.append(np.random.normal(0.0005, 0.02) + momentum)

    momentum_returns = np.array(momentum_returns)

    # Test for normality
    efficient_normal = stats.shapiro(efficient_returns)
    momentum_normal = stats.shapiro(momentum_returns)

    print(f"Efficient Market Normality Test: p-value = {efficient_normal.pvalue".4f"}")
    print(f"Momentum Market Normality Test: p-value = {momentum_normal.pvalue".4f"}")

    # Test for autocorrelation
    efficient_autocorr = stats.pearsonr(efficient_returns[:-1], efficient_returns[1:])
    momentum_autocorr = stats.pearsonr(momentum_returns[:-1], momentum_returns[1:])

    print(f"Efficient Market Autocorrelation: r = {efficient_autocorr[0]".4f"}")
    print(f"Momentum Market Autocorrelation: r = {momentum_autocorr[0]".4f"}")

    # Variance comparison
    variance_test = stats.levene(efficient_returns, momentum_returns)
    print(f"Variance Equality Test: p-value = {variance_test.pvalue".4f"}")

    print()
```

### **Example 3: Monte Carlo Risk Simulation**
```python
def monte_carlo_risk_simulation():
    """Monte Carlo simulation for risk analysis."""
    print("=== Monte Carlo Risk Simulation ===")

    # Portfolio parameters
    initial_value = 1000000
    expected_return = 0.08
    volatility = 0.15
    time_horizon = 1  # 1 year
    n_simulations = 10000
    confidence_level = 0.95

    # Run Monte Carlo simulation
    np.random.seed(42)
    simulated_values = []

    for _ in range(n_simulations):
        # Simulate annual return
        annual_return = np.random.normal(expected_return, volatility)
        final_value = initial_value * (1 + annual_return)
        simulated_values.append(final_value)

    simulated_values = np.array(simulated_values)

    # Calculate risk metrics
    final_mean = np.mean(simulated_values)
    final_std = np.std(simulated_values)
    var_95 = np.percentile(simulated_values, (1 - confidence_level) * 100)
    cvar_95 = np.mean(simulated_values[simulated_values <= var_95])
    max_drawdown = (initial_value - np.min(simulated_values)) / initial_value

    print(f"Expected Final Value: ${final_mean",.0f"}")
    print(f"Value at Risk (95%): ${var_95",.0f"}")
    print(f"Conditional VaR (95%): ${cvar_95",.0f"}")
    print(f"Maximum Drawdown: {max_drawdown".2%"}")
    print(f"Probability of Loss: {np.mean(simulated_values < initial_value)".2%"}")

    # Generate confidence intervals
    ci_lower = np.percentile(simulated_values, 2.5)
    ci_upper = np.percentile(simulated_values, 97.5)
    print(f"95% Confidence Interval: ${ci_lower",.0f"} - ${ci_upper",.0f"}")

    print()
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python -m pytest tests/test_statistics.py -v
```

## 📚 References

- [Statistics for Finance](https://www.oreilly.com/library/view/statistics-for-finance/9781482298019/)
- [Quantitative Finance Stack Exchange](https://quant.stackexchange.com/)
- [SciPy Statistics Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Risk Management and Financial Institutions](https://www.wiley.com/en-us/Risk+Management+and+Financial+Institutions%2C+5th+Edition-p-9781119448112)

## 🎓 Learning Path

### **Prerequisites**
- Basic probability and statistics
- Understanding of financial markets

### **Next Steps**
- **Regression Analysis**: Linear and nonlinear modeling
- **Time Series Analysis**: ARIMA, GARCH models
- **Machine Learning**: Advanced predictive modeling

### **Assessment**
1. Calculate Value at Risk for a portfolio using different methods
2. Perform hypothesis testing to compare two investment strategies
3. Build a Monte Carlo simulation for option pricing
4. Analyze the correlation structure of a multi-asset portfolio

---

*This utility provides essential statistical tools for quantitative finance. Master these techniques to understand risk, measure performance, and make data-driven investment decisions.*
