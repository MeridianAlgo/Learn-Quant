"""
Options Greeks Calculator - Delta, Gamma, Theta, Vega, Rho
"""

import math


def calculate_d1_d2(S, K, T, r, sigma):
    """Calculate d1 and d2 for Black-Scholes formula."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def norm_cdf(x):
    """Cumulative distribution function for standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x):
    """Probability density function for standard normal distribution."""
    return math.exp(-(x**2) / 2.0) / math.sqrt(2.0 * math.pi)


def calculate_delta(S, K, T, r, sigma, option_type="call"):
    """
    Delta: Rate of change of option price with respect to underlying price
    Call Delta: 0 to 1
    Put Delta: -1 to 0
    """
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return norm_cdf(d1)
    else:  # put
        return norm_cdf(d1) - 1


def calculate_gamma(S, K, T, r, sigma):
    """
    Gamma: Rate of change of delta with respect to underlying price
    Same for calls and puts
    """
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))


def calculate_theta(S, K, T, r, sigma, option_type="call"):
    """
    Theta: Rate of change of option price with respect to time
    Usually negative (time decay)
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)

    term1 = -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))

    if option_type == "call":
        term2 = r * K * math.exp(-r * T) * norm_cdf(d2)
        return (term1 - term2) / 365  # Per day
    else:  # put
        term2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
        return (term1 + term2) / 365  # Per day


def calculate_vega(S, K, T, r, sigma):
    """
    Vega: Rate of change of option price with respect to volatility
    Same for calls and puts
    """
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return S * norm_pdf(d1) * math.sqrt(T) / 100  # Per 1% change in volatility


def calculate_rho(S, K, T, r, sigma, option_type="call"):
    """
    Rho: Rate of change of option price with respect to interest rate
    """
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return K * T * math.exp(-r * T) * norm_cdf(d2) / 100  # Per 1% change in rate
    else:  # put
        return -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100


def calculate_all_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculate all Greeks at once."""
    return {
        "delta": calculate_delta(S, K, T, r, sigma, option_type),
        "gamma": calculate_gamma(S, K, T, r, sigma),
        "theta": calculate_theta(S, K, T, r, sigma, option_type),
        "vega": calculate_vega(S, K, T, r, sigma),
        "rho": calculate_rho(S, K, T, r, sigma, option_type),
    }


if __name__ == "__main__":
    # Example: Calculate Greeks for an option
    S = 100  # Stock price
    K = 100  # Strike price
    T = 0.25  # Time to expiration (3 months)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.20  # Volatility (20%)

    print("=== Options Greeks Calculator ===\n")
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T*365:.0f} days")
    print(f"Risk-Free Rate: {r*100:.1f}%")
    print(f"Volatility: {sigma*100:.1f}%\n")

    # Call Greeks
    call_greeks = calculate_all_greeks(S, K, T, r, sigma, "call")
    print("CALL OPTION GREEKS:")
    print(f"  Delta: {call_greeks['delta']:.4f} (price change per $1 move)")
    print(f"  Gamma: {call_greeks['gamma']:.4f} (delta change per $1 move)")
    print(f"  Theta: {call_greeks['theta']:.4f} (price change per day)")
    print(f"  Vega: {call_greeks['vega']:.4f} (price change per 1% vol)")
    print(f"  Rho: {call_greeks['rho']:.4f} (price change per 1% rate)\n")

    # Put Greeks
    put_greeks = calculate_all_greeks(S, K, T, r, sigma, "put")
    print("PUT OPTION GREEKS:")
    print(f"  Delta: {put_greeks['delta']:.4f} (price change per $1 move)")
    print(f"  Gamma: {put_greeks['gamma']:.4f} (delta change per $1 move)")
    print(f"  Theta: {put_greeks['theta']:.4f} (price change per day)")
    print(f"  Vega: {put_greeks['vega']:.4f} (price change per 1% vol)")
    print(f"  Rho: {put_greeks['rho']:.4f} (price change per 1% rate)")
