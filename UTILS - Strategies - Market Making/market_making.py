"""
Avellaneda-Stoikov Market Making Model
----------------------------------------
A continuous-time model for optimal bid-ask spread setting.
The market maker maximizes expected PnL while managing inventory risk.

Key concepts:
- Reservation price: Mid-price adjusted for inventory position
- Optimal spread: Bid-ask spread balancing order flow vs inventory risk
"""

import numpy as np
from typing import Optional


def reservation_price(
    mid_price: float,
    inventory: float,
    T: float,
    t: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Market maker's reservation (indifference) price.
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    Args:
        mid_price: Current mid price.
        inventory: Current inventory (positive = long, negative = short).
        T: Trading horizon (e.g., 1.0 for end of day).
        t: Current time (fraction of T elapsed).
        sigma: Asset volatility per unit time.
        gamma: Risk aversion coefficient.

    Returns:
        float: Reservation price.
    """
    return float(mid_price - inventory * gamma * sigma**2 * (T - t))


def optimal_spread(
    T: float,
    t: float,
    sigma: float,
    gamma: float,
    kappa: float,
) -> float:
    """
    Avellaneda-Stoikov optimal bid-ask spread.
    spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)

    Args:
        T: Trading horizon.
        t: Current time.
        sigma: Asset volatility.
        gamma: Risk aversion.
        kappa: Order arrival intensity parameter.

    Returns:
        float: Optimal total spread.
    """
    return float(gamma * sigma**2 * (T - t) + (2 / gamma) * np.log(1 + gamma / kappa))


def bid_ask_quotes(
    mid_price: float,
    inventory: float,
    T: float,
    t: float,
    sigma: float,
    gamma: float,
    kappa: float,
) -> dict:
    """
    Compute optimal bid and ask quotes.

    Returns:
        dict: bid, ask, reservation_price, spread.
    """
    r = reservation_price(mid_price, inventory, T, t, sigma, gamma)
    spread = optimal_spread(T, t, sigma, gamma, kappa)
    half_spread = spread / 2
    return {
        "bid": r - half_spread,
        "ask": r + half_spread,
        "reservation_price": r,
        "spread": spread,
    }


def simulate_market_maker(
    S0: float = 100.0,
    sigma: float = 2.0,
    gamma: float = 0.1,
    kappa: float = 1.5,
    T: float = 1.0,
    dt: float = 0.005,
    n_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate a market maker using the Avellaneda-Stoikov model.

    Args:
        S0: Initial mid price.
        sigma: Volatility.
        gamma: Risk aversion.
        kappa: Order arrival intensity.
        T: Total time horizon.
        dt: Time step.
        n_steps: Override number of steps (optional).
        seed: Random seed.

    Returns:
        dict: time, mid_price, inventory, cash, pnl, bids, asks arrays.
    """
    if seed is not None:
        np.random.seed(seed)
    if n_steps is None:
        n_steps = int(T / dt)

    times = np.linspace(0, T, n_steps)
    S = np.zeros(n_steps)
    S[0] = S0
    inventory = np.zeros(n_steps)
    cash = np.zeros(n_steps)
    bids = np.zeros(n_steps)
    asks = np.zeros(n_steps)

    for i in range(1, n_steps):
        t = times[i - 1]
        q = inventory[i - 1]
        s = S[i - 1]

        S[i] = s + sigma * np.random.randn() * np.sqrt(dt)

        quotes = bid_ask_quotes(s, q, T, t, sigma, gamma, kappa)
        bids[i] = quotes["bid"]
        asks[i] = quotes["ask"]

        bid_fill = np.random.poisson(kappa * np.exp(-kappa * (s - quotes["bid"])) * dt)
        ask_fill = np.random.poisson(kappa * np.exp(-kappa * (quotes["ask"] - s)) * dt)

        inventory[i] = q + bid_fill - ask_fill
        cash[i] = cash[i - 1] + ask_fill * quotes["ask"] - bid_fill * quotes["bid"]

    pnl = cash + inventory * S - (cash[0] + inventory[0] * S[0])

    return {
        "time": times,
        "mid_price": S,
        "inventory": inventory,
        "cash": cash,
        "pnl": pnl,
        "bids": bids,
        "asks": asks,
    }


if __name__ == "__main__":
    np.random.seed(42)
    result = simulate_market_maker(S0=100, sigma=2.0, gamma=0.1, kappa=1.5, seed=42)

    print("Avellaneda-Stoikov Market Making Simulation")
    print("=" * 50)
    print(f"Final PnL:          {result['pnl'][-1]:.2f}")
    print(f"Final Inventory:    {result['inventory'][-1]:.0f}")
    print(f"Max |Inventory|:    {np.max(np.abs(result['inventory'])):.0f}")

    quotes = bid_ask_quotes(100, 0, 1.0, 0.25, 2.0, 0.1, 1.5)
    print(f"\nSample quotes (q=0, t=0.25):")
    print(f"  Bid: {quotes['bid']:.3f}  Ask: {quotes['ask']:.3f}  Spread: {quotes['spread']:.3f}")
