import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Strategies - Market Making"))
from market_making import bid_ask_quotes, optimal_spread, reservation_price, simulate_market_maker


def test_reservation_price_no_inventory():
    r = reservation_price(100.0, 0, T=1.0, t=0.5, sigma=2.0, gamma=0.1)
    assert r == 100.0


def test_reservation_price_long_inventory_lower():
    """Long inventory -> reservation price below mid."""
    r = reservation_price(100.0, inventory=10, T=1.0, t=0.0, sigma=2.0, gamma=0.1)
    assert r < 100.0


def test_reservation_price_short_inventory_higher():
    """Short inventory -> reservation price above mid."""
    r = reservation_price(100.0, inventory=-10, T=1.0, t=0.0, sigma=2.0, gamma=0.1)
    assert r > 100.0


def test_optimal_spread_positive():
    spread = optimal_spread(T=1.0, t=0.0, sigma=2.0, gamma=0.1, kappa=1.5)
    assert spread > 0


def test_optimal_spread_decreases_over_time():
    """Spread should narrow as trading horizon approaches end."""
    spread_early = optimal_spread(T=1.0, t=0.0, sigma=2.0, gamma=0.1, kappa=1.5)
    spread_late = optimal_spread(T=1.0, t=0.9, sigma=2.0, gamma=0.1, kappa=1.5)
    assert spread_late < spread_early


def test_bid_below_ask():
    quotes = bid_ask_quotes(100, 0, T=1.0, t=0.0, sigma=2.0, gamma=0.1, kappa=1.5)
    assert quotes["bid"] < quotes["ask"]


def test_bid_ask_quotes_keys():
    quotes = bid_ask_quotes(100, 0, T=1.0, t=0.5, sigma=2.0, gamma=0.1, kappa=1.5)
    assert set(quotes.keys()) == {"bid", "ask", "reservation_price", "spread"}


def test_simulation_returns_correct_keys():
    result = simulate_market_maker(S0=100, n_steps=50, seed=42)
    for key in ["time", "mid_price", "inventory", "cash", "pnl", "bids", "asks"]:
        assert key in result


def test_simulation_array_lengths():
    n = 50
    result = simulate_market_maker(S0=100, T=1.0, dt=1.0 / n, n_steps=n, seed=42)
    assert len(result["mid_price"]) == n
