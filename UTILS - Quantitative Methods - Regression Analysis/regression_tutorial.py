"""Regression Analysis for Quantitative Finance.

Run with:
    python regression_tutorial.py

This module teaches linear regression, multiple regression, and financial applications.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("QUANTITATIVE METHODS – REGRESSION ANALYSIS")
    print("#" * 60)
    print("Linear regression, beta calculation, factor models,")
    print("and predictive modeling for finance.\n")


def simple_linear_regression() -> None:
    """Demonstrate simple linear regression."""
    print("=" * 60)
    print("SIMPLE LINEAR REGRESSION")
    print("=" * 60)

    # Example: Stock returns vs market returns (calculating beta)
    np.random.seed(42)

    # Generate market returns
    market_returns = np.random.normal(0.001, 0.02, 252)

    # Generate stock returns (correlated with market)
    beta = 1.3
    alpha = 0.0005
    stock_returns = alpha + beta * market_returns + np.random.normal(0, 0.01, 252)

    # Calculate regression: stock_returns = alpha + beta * market_returns
    # Using numpy polyfit (degree 1 = linear)
    coefficients = np.polyfit(market_returns, stock_returns, deg=1)
    beta_estimated = coefficients[0]
    alpha_estimated = coefficients[1]

    print("\nRegression: Stock Returns = α + β × Market Returns")
    print(f"  True Beta: {beta:.3f}")
    print(f"  Estimated Beta: {beta_estimated:.3f}")
    print(f"  True Alpha: {alpha:.5f}")
    print(f"  Estimated Alpha: {alpha_estimated:.5f}")

    # Calculate R-squared
    y_pred = alpha_estimated + beta_estimated * market_returns
    ss_res = np.sum((stock_returns - y_pred) ** 2)
    ss_tot = np.sum((stock_returns - np.mean(stock_returns)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"  R-squared: {r_squared:.4f} ({r_squared * 100:.2f}% variance explained)")


def multiple_regression() -> None:
    """Demonstrate multiple linear regression."""
    print("\n" + "=" * 60)
    print("MULTIPLE LINEAR REGRESSION")
    print("=" * 60)

    # Fama-French 3-factor model example
    # Stock Returns = α + β₁(Market) + β₂(SMB) + β₃(HML) + ε

    np.random.seed(42)
    n_periods = 252

    # Generate factor returns
    market_factor = np.random.normal(0.001, 0.02, n_periods)
    smb_factor = np.random.normal(0.0005, 0.015, n_periods)  # Small minus Big
    hml_factor = np.random.normal(0.0003, 0.012, n_periods)  # High minus Low

    # True coefficients
    alpha_true = 0.0002
    beta_market = 1.2
    beta_smb = 0.5
    beta_hml = -0.3

    # Generate stock returns
    stock_returns = (
        alpha_true
        + beta_market * market_factor
        + beta_smb * smb_factor
        + beta_hml * hml_factor
        + np.random.normal(0, 0.008, n_periods)
    )

    # Create design matrix X
    # X has shape (n_periods, 4): [1, market, smb, hml]
    X = np.column_stack([np.ones(n_periods), market_factor, smb_factor, hml_factor])

    # Solve using normal equation: β = (X^T X)^(-1) X^T y
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ stock_returns

    alpha_est, beta_market_est, beta_smb_est, beta_hml_est = coefficients

    print("\n3-Factor Model: Returns = α + β₁(Market) + β₂(SMB) + β₃(HML)")
    print("\n" + "-" * 60)
    print(f"{'Factor':<15} {'True':<12} {'Estimated':<12}")
    print("-" * 60)
    print(f"{'Alpha':<15} {alpha_true:>11.5f} {alpha_est:>11.5f}")
    print(f"{'Market Beta':<15} {beta_market:>11.3f} {beta_market_est:>11.3f}")
    print(f"{'SMB Beta':<15} {beta_smb:>11.3f} {beta_smb_est:>11.3f}")
    print(f"{'HML Beta':<15} {beta_hml:>11.3f} {betahml_est:>11.3f}")
    print("-" * 60)

    # Calculate R-squared
    y_pred = X @ coefficients
    ss_res = np.sum((stock_returns - y_pred) ** 2)
    ss_tot = np.sum((stock_returns - np.mean(stock_returns)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nR-squared: {r_squared:.4f}")


def beta_calculation() -> None:
    """Demonstrate beta calculation using regression."""
    print("\n" + "=" * 60)
    print("BETA CALCULATION")
    print("=" * 60)

    # Simulate returns for 5 stocks vs market
    np.random.seed(42)
    n_periods = 252

    market_returns = np.random.normal(0.001, 0.02, n_periods)

    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "DIS"]
    true_betas = [1.2, 1.1, 0.9, 1.8, 1.0]
    true_alphas = [0.0005, 0.0003, 0.0002, 0.001, 0.0001]

    print("\nCalculating Beta for Multiple Stocks:")
    print("\n" + "-" * 70)
    print(f"{'Ticker':<10} {'True Beta':<12} {'Est. Beta':<12} {'Alpha':<12} {'R²':<10}")
    print("-" * 70)

    for ticker, true_beta, true_alpha in zip(tickers, true_betas, true_alphas):
        # Generate stock returns
        stock_returns = true_alpha + true_beta * market_returns + np.random.normal(0, 0.01, n_periods)

        # Calculate beta using covariance method
        # β = Cov(stock, market) / Var(market)
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta_cov = covariance / market_variance

        # Calculate using regression
        coeffs = np.polyfit(market_returns, stock_returns, deg=1)
        beta_reg = coeffs[0]
        alpha_reg = coeffs[1]

        # R-squared
        y_pred = alpha_reg + beta_reg * market_returns
        ss_res = np.sum((stock_returns - y_pred) ** 2)
        ss_tot = np.sum((stock_returns - np.mean(stock_returns)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"{ticker:<10} {true_beta:>11.3f} {beta_reg:>11.3f} {alpha_reg:>11.5f} {r_squared:>9.4f}")

    print("-" * 70)


def regression_diagnostics() -> None:
    """Demonstrate regression diagnostics."""
    print("\n" + "=" * 60)
    print("REGRESSION DIAGNOSTICS")
    print("=" * 60)

    np.random.seed(42)

    # Generate data
    x = np.random.normal(0, 1, 100)
    y = 2 + 3 * x + np.random.normal(0, 0.5, 100)

    # Fit regression
    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = coeffs[0], coeffs[1]

    # Predictions
    y_pred = intercept + slope * x

    # Residuals
    residuals = y - y_pred

    # Standard Error of Residuals
    n = len(x)
    se_residuals = np.sqrt(np.sum(residuals**2) / (n - 2))

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Adjusted R-squared (for multiple regression)
    k = 1  # number of predictors
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))

    # Standard errors of coefficients
    x_mean = np.mean(x)
    se_slope = se_residuals / np.sqrt(np.sum((x - x_mean) ** 2))
    se_intercept = se_residuals * np.sqrt(1 / n + x_mean**2 / np.sum((x - x_mean) ** 2))

    # T-statistics
    t_slope = slope / se_slope
    t_intercept = intercept / se_intercept

    print("\nRegression Results:")
    print("-" * 60)
    print(f"Slope (β₁): {slope:.4f} ± {se_slope:.4f} (t = {t_slope:.2f})")
    print(f"Intercept (β₀): {intercept:.4f} ± {se_intercept:.4f} (t = {t_intercept:.2f})")
    print(f"\nR-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {adj_r_squared:.4f}")
    print(f"Standard Error: {se_residuals:.4f}")
    print(f"Observations: {n}")
    print("-" * 60)

    # Residual analysis
    print("\nResidual Analysis:")
    print(f"  Mean of residuals: {np.mean(residuals):.6f} (should be ~0)")
    print(f"  Std of residuals: {np.std(residuals):.4f}")
    print(f"  Min residual: {np.min(residuals):.4f}")
    print(f"  Max residual: {np.max(residuals):.4f}")


def practical_example_market_model() -> None:
    """Practical example: Market model for portfolio."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLE: MARKET MODEL")
    print("=" * 60)

    # Simulate portfolio returns vs market
    np.random.seed(42)
    n_months = 60  # 5 years of monthly data

    # Market returns (S&P 500)
    market_returns = np.random.normal(0.01, 0.045, n_months)

    # Portfolio returns
    portfolio_alpha = 0.002  # 0.2% monthly alpha
    portfolio_beta = 1.15  # 15% more volatile than market
    portfolio_returns = portfolio_alpha + portfolio_beta * market_returns + np.random.normal(0, 0.02, n_months)

    # Regression analysis
    coeffs = np.polyfit(market_returns, portfolio_returns, deg=1)
    beta_est = coeffs[0]
    alpha_est = coeffs[1]

    # Calculate metrics
    y_pred = alpha_est + beta_est * market_returns
    residuals = portfolio_returns - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((portfolio_returns - np.mean(portfolio_returns)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Annualized metrics
    alpha_annual = alpha_est * 12

    # Information Ratio (alpha / tracking error)
    tracking_error = np.std(residuals) * np.sqrt(12)
    information_ratio = alpha_annual / tracking_error if tracking_error > 0 else 0

    print("\nMarket Model: Portfolio Returns = α + β × Market Returns")
    print("\n" + "=" * 60)
    print(f"Beta: {beta_est:.3f}")
    print(f"  → Portfolio is {abs(beta_est - 1) * 100:.1f}% {'more' if beta_est > 1 else 'less'} volatile than market")

    print(f"\nAlpha (Monthly): {alpha_est:.4f} ({alpha_est * 100:.2f}%)")
    print(f"Alpha (Annual): {alpha_annual:.4f} ({alpha_annual * 100:.2f}%)")
    print(
        f"  → Portfolio {'outperforms' if alpha_est > 0 else 'underperforms'} market by {abs(alpha_annual) * 100:.2f}% annually"
    )

    print(f"\nR-squared: {r_squared:.4f}")
    print(f"  → {r_squared * 100:.1f}% of portfolio variance explained by market")

    print(f"\nTracking Error (Annual): {tracking_error:.2%}")
    print(f"Information Ratio: {information_ratio:.3f}")
    print("=" * 60)

    # Interpretation
    print("\nInterpretation:")
    if beta_est > 1.1:
        print("  → Aggressive portfolio (high market sensitivity)")
    elif beta_est < 0.9:
        print("  → Defensive portfolio (low market sensitivity)")
    else:
        print("  → Neutral portfolio (close to market)")

    if alpha_est > 0.001:
        print("  → Positive alpha suggests skill/outperformance")
    elif alpha_est < -0.001:
        print("  → Negative alpha suggests underperformance")
    else:
        print("  → Alpha near zero suggests market-like performance")


def main() -> None:
    """Run all regression examples."""
    intro()
    simple_linear_regression()
    multiple_regression()
    beta_calculation()
    regression_diagnostics()
    practical_example_market_model()
    print("\n🎉 Regression Analysis tutorial complete!")
    print("Use regression for beta calculation, factor models, and performance attribution.")


if __name__ == "__main__":
    main()
