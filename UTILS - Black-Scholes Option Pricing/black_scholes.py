"""
Black-Scholes Option Pricing Model
------------------------------------
This utility helps calculate the fair price of a European call or put option using the Black-Scholes formula.
"""

import numpy as np
import scipy.stats as stats


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes price for a European option.
    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to maturity (in years, e.g., 0.5 for 6 months)
        r (float): Risk-free annual interest rate (as decimal)
        sigma (float): Volatility of the stock (annualized, as decimal)
        option_type (str): 'call' for call option, 'put' for put option
    Returns:
        float: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return price


if __name__ == "__main__":
    # Example usage: Priced with fictional market data
    S = 100  # Stock price
    K = 105  # Strike price
    T = 1  # 1 year to expiry
    r = 0.03  # 3% risk-free rate
    sigma = 0.2  # 20% volatility
    price_call = black_scholes(S, K, T, r, sigma, "call")
    price_put = black_scholes(S, K, T, r, sigma, "put")
    print(f"Call price: {price_call:.2f}")
    print(f"Put price: {price_put:.2f}")
