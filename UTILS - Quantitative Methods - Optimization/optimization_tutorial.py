"""Optimization Tutorial for Quantitative Finance.

Run with:
    python optimization_tutorial.py

This module teaches mathematical optimization using scipy.optimize,
focusing on portfolio optimization and curve fitting.
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("QUANTITATIVE METHODS – OPTIMIZATION")
    print("#" * 60)
    print("Learn to minimize functions, optimize portfolios,")
    print("and fit models to financial data.\n")


def basic_minimization() -> None:
    """Demonstrate basic function minimization."""
    print("=" * 60)
    print("BASIC FUNCTION MINIMIZATION")
    print("=" * 60)
    
    # Example: Find x that minimizes f(x) = (x - 3)^2 + 5
    # Minimum should be at x = 3, value = 5
    
    def objective_function(x):
        return (x[0] - 3)**2 + 5
    
    # Initial guess
    x0 = [0.0]
    
    # Minimize
    result = minimize(objective_function, x0, method='BFGS')
    
    print(f"Function: f(x) = (x - 3)² + 5")
    print(f"Initial guess: x = {x0[0]}")
    print(f"Optimal x: {result.x[0]:.4f}")
    print(f"Minimum value: {result.fun:.4f}")
    print(f"Success: {result.success}")


def portfolio_optimization() -> None:
    """Demonstrate portfolio optimization (Markowitz)."""
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION (Mean-Variance)")
    print("=" * 60)
    
    # Simulated data
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    n_assets = len(tickers)
    
    # Expected returns
    expected_returns = np.array([0.12, 0.15, 0.10, 0.14])
    
    # Covariance matrix
    cov_matrix = np.array([
        [0.040, 0.015, 0.018, 0.020],
        [0.015, 0.050, 0.012, 0.025],
        [0.018, 0.012, 0.035, 0.015],
        [0.020, 0.025, 0.015, 0.055]
    ])
    
    print("Assets:", tickers)
    print("Expected Returns:", expected_returns)
    
    # 1. Minimize Volatility (Global Minimum Variance Portfolio)
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: 0 <= weight <= 1 (Long only)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess (equal weights)
    init_guess = n_assets * [1. / n_assets]
    
    # Optimize
    result = minimize(portfolio_volatility, init_guess, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    print("\nGlobal Minimum Variance Portfolio:")
    print("-" * 30)
    for ticker, weight in zip(tickers, result.x):
        print(f"{ticker}: {weight:.2%}")
    
    print(f"Portfolio Volatility: {result.fun:.2%}")
    print(f"Expected Return: {np.dot(result.x, expected_returns):.2%}")
    
    # 2. Maximize Sharpe Ratio (Tangency Portfolio)
    # Minimize negative Sharpe Ratio
    risk_free_rate = 0.02
    
    def negative_sharpe(weights):
        p_ret = np.dot(weights, expected_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - risk_free_rate) / p_vol
    
    result_sharpe = minimize(negative_sharpe, init_guess,
                            method='SLSQP', bounds=bounds, constraints=constraints)
    
    print("\nMaximum Sharpe Ratio Portfolio:")
    print("-" * 30)
    for ticker, weight in zip(tickers, result_sharpe.x):
        print(f"{ticker}: {weight:.2%}")
    
    max_sharpe_ret = np.dot(result_sharpe.x, expected_returns)
    max_sharpe_vol = portfolio_volatility(result_sharpe.x)
    
    print(f"Portfolio Volatility: {max_sharpe_vol:.2%}")
    print(f"Expected Return: {max_sharpe_ret:.2%}")
    print(f"Sharpe Ratio: {-result_sharpe.fun:.4f}")


def curve_fitting() -> None:
    """Demonstrate curve fitting (Yield Curve example)."""
    print("\n" + "=" * 60)
    print("CURVE FITTING (Nelson-Siegel Model)")
    print("=" * 60)
    
    # Nelson-Siegel Yield Curve Model
    # y(t) = β0 + β1*((1-exp(-t/τ))/(t/τ)) + β2*(((1-exp(-t/τ))/(t/τ)) - exp(-t/τ))
    
    def nelson_siegel(t, beta0, beta1, beta2, tau):
        term1 = (1 - np.exp(-t/tau)) / (t/tau)
        term2 = term1 - np.exp(-t/tau)
        return beta0 + beta1 * term1 + beta2 * term2
    
    # Simulated market data (Maturity in years, Yield in %)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    market_yields = np.array([5.25, 5.30, 5.35, 5.20, 5.10, 5.05, 5.15, 5.25, 5.50, 5.60])
    
    # Fit model to data
    # Initial guess: β0=5, β1=-1, β2=1, τ=2
    p0 = [5.0, -1.0, 1.0, 2.0]
    
    try:
        params, covariance = curve_fit(nelson_siegel, maturities, market_yields, p0=p0)
        
        beta0, beta1, beta2, tau = params
        
        print("Calibrated Nelson-Siegel Parameters:")
        print(f"  β0 (Long-term): {beta0:.4f}")
        print(f"  β1 (Short-term): {beta1:.4f}")
        print(f"  β2 (Curvature): {beta2:.4f}")
        print(f"  τ (Decay): {tau:.4f}")
        
        # Calculate fitted yields
        fitted_yields = nelson_siegel(maturities, *params)
        
        print("\nModel Fit:")
        print(f"{'Maturity':<10} {'Market':<10} {'Model':<10} {'Error':<10}")
        print("-" * 45)
        for m, y_mkt, y_mod in zip(maturities, market_yields, fitted_yields):
            error = y_mod - y_mkt
            print(f"{m:<10.2f} {y_mkt:<10.2f} {y_mod:<10.2f} {error:<10.4f}")
            
        rmse = np.sqrt(np.mean((fitted_yields - market_yields)**2))
        print(f"\nRoot Mean Squared Error: {rmse:.4f}%")
        
    except Exception as e:
        print(f"Optimization failed: {e}")


def root_finding() -> None:
    """Demonstrate root finding (Implied Volatility)."""
    print("\n" + "=" * 60)
    print("ROOT FINDING (Implied Volatility)")
    print("=" * 60)
    
    from scipy.optimize import newton
    from scipy.stats import norm
    
    # Black-Scholes Call Price
    def bs_call(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    # Market parameters
    S = 100.0   # Spot price
    K = 105.0   # Strike price
    T = 1.0     # Time to maturity
    r = 0.05    # Risk-free rate
    market_price = 8.50  # Observed option price
    
    print(f"Market Call Price: ${market_price:.2f}")
    print(f"Parameters: S={S}, K={K}, T={T}, r={r}")
    
    # Objective: Find sigma such that bs_call(sigma) - market_price = 0
    def objective(sigma):
        return bs_call(S, K, T, r, sigma) - market_price
    
    try:
        # Newton-Raphson method
        implied_vol = newton(objective, x0=0.2)
        
        print(f"\nCalculated Implied Volatility: {implied_vol:.2%}")
        
        # Verify
        price_check = bs_call(S, K, T, r, implied_vol)
        print(f"Price with IV: ${price_check:.2f} (Diff: ${price_check - market_price:.4f})")
        
    except Exception as e:
        print(f"Root finding failed: {e}")


def main() -> None:
    """Run all optimization examples."""
    intro()
    basic_minimization()
    portfolio_optimization()
    curve_fitting()
    root_finding()
    print("\n🎉 Optimization tutorial complete!")
    print("Optimization is the engine behind modern portfolio theory and calibration.")


if __name__ == "__main__":
    main()
