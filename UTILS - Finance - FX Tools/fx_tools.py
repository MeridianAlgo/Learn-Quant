"""
FX (Foreign Exchange) Tools
-----------------------------
Key FX analytics for understanding exchange rate dynamics.

- Forward rates: No-arbitrage forward rates via Interest Rate Parity
- CIP deviation: Covered Interest Parity basis
- Cross rates: Deriving exchange rates from two pairs
- Garman-Kohlhagen: European FX option pricing
- Triangular arbitrage detection
"""

import numpy as np
import scipy.stats as stats


def forward_rate(
    spot: float,
    r_domestic: float,
    r_foreign: float,
    T: float,
) -> float:
    """
    No-arbitrage forward exchange rate via Covered Interest Rate Parity.
    F = S * exp((r_d - r_f) * T)

    Args:
        spot: Spot exchange rate (domestic per foreign, e.g., USD/EUR).
        r_domestic: Domestic risk-free rate (continuously compounded).
        r_foreign: Foreign risk-free rate (continuously compounded).
        T: Time to delivery (years).

    Returns:
        float: Forward exchange rate.
    """
    return float(spot * np.exp((r_domestic - r_foreign) * T))


def forward_points(
    spot: float,
    r_domestic: float,
    r_foreign: float,
    T: float,
    pip_size: float = 0.0001,
) -> float:
    """
    Forward points = (Forward - Spot) in pips.

    Returns:
        float: Forward points.
    """
    fwd = forward_rate(spot, r_domestic, r_foreign, T)
    return float((fwd - spot) / pip_size)


def cip_deviation(
    spot: float,
    forward: float,
    r_domestic: float,
    r_foreign: float,
    T: float,
) -> float:
    """
    Covered Interest Parity deviation in basis points.
    CIP holds when F = S * exp((r_d - r_f) * T).
    Deviation = (implied r_d - actual r_d) * 10,000.

    Args:
        spot: Spot rate.
        forward: Observed market forward rate.
        r_domestic: Domestic rate.
        r_foreign: Foreign rate.
        T: Tenor (years).

    Returns:
        float: CIP deviation in basis points.
    """
    implied_r_d = np.log(forward / spot) / T + r_foreign
    return float((implied_r_d - r_domestic) * 10_000)


def cross_rate(s_ab: float, s_ac: float) -> float:
    """
    Cross rate B/C from A/B and A/C quotes.
    e.g., given USD/EUR and USD/GBP, compute EUR/GBP.

    Args:
        s_ab: Spot A/B (units of A per B).
        s_ac: Spot A/C (units of A per C).

    Returns:
        float: Cross rate B/C.
    """
    return float(s_ac / s_ab)


def triangular_arbitrage_profit(
    s_ab: float,
    s_bc: float,
    s_ca: float,
    notional: float = 1_000_000,
) -> float:
    """
    Detect triangular arbitrage: A → B → C → A.
    Profit = (S_AB * S_BC * S_CA - 1) * notional.

    Args:
        s_ab: A→B rate.
        s_bc: B→C rate.
        s_ca: C→A rate.
        notional: Starting position size.

    Returns:
        float: Profit per notional (positive = arbitrage exists).
    """
    return float((s_ab * s_bc * s_ca - 1) * notional)


def garman_kohlhagen(
    S: float,
    K: float,
    r_d: float,
    r_f: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> dict:
    """
    Garman-Kohlhagen formula for European FX options.
    Extension of Black-Scholes where foreign rate acts as a continuous dividend.

    Args:
        S: Spot exchange rate.
        K: Strike.
        r_d: Domestic risk-free rate.
        r_f: Foreign risk-free rate.
        sigma: FX volatility.
        T: Time to expiry (years).
        option_type: "call" or "put".

    Returns:
        dict: price, delta, gamma, vega (per 1% vol), theta (per day).
    """
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    disc_d = np.exp(-r_d * T)
    disc_f = np.exp(-r_f * T)

    if option_type == "call":
        price = S * disc_f * stats.norm.cdf(d1) - K * disc_d * stats.norm.cdf(d2)
        delta = float(disc_f * stats.norm.cdf(d1))
    else:
        price = K * disc_d * stats.norm.cdf(-d2) - S * disc_f * stats.norm.cdf(-d1)
        delta = float(-disc_f * stats.norm.cdf(-d1))

    gamma = float(disc_f * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega = float(S * disc_f * stats.norm.pdf(d1) * np.sqrt(T) / 100)

    if option_type == "call":
        theta = float((
            -S * disc_f * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r_d * K * disc_d * stats.norm.cdf(d2)
            + r_f * S * disc_f * stats.norm.cdf(d1)
        ) / 365)
    else:
        theta = float((
            -S * disc_f * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r_d * K * disc_d * stats.norm.cdf(-d2)
            - r_f * S * disc_f * stats.norm.cdf(-d1)
        ) / 365)

    return {
        "price": float(price),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


if __name__ == "__main__":
    print("FX Tools")
    print("=" * 40)

    spot, r_d, r_f = 1.1000, 0.05, 0.02

    fwd_1y = forward_rate(spot, r_d, r_f, 1.0)
    print(f"\nSpot USD/EUR:        {spot:.4f}")
    print(f"1Y Forward:          {fwd_1y:.4f}")
    print(f"Forward points:      {forward_points(spot, r_d, r_f, 1.0):.1f} pips")

    # Simulate a small forward mispricing
    dev = cip_deviation(spot, fwd_1y * 1.001, r_d, r_f, 1.0)
    print(f"CIP deviation:       {dev:.2f} bps")

    call = garman_kohlhagen(spot, 1.10, r_d, r_f, 0.10, 0.25)
    print(f"\nGarman-Kohlhagen 3M ATM Call (vol=10%):")
    print(f"  Price: {call['price']:.4f}  Delta: {call['delta']:.4f}  Vega: {call['vega']:.4f}")

    profit = triangular_arbitrage_profit(1.30, 0.86, 0.89)
    print(f"\nTriangular arb profit: ${profit:.2f}")
