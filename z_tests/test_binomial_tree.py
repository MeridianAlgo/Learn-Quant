import sys
from math import erf, exp, log, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Options Pricing - Binomial Tree"))
from binomial_tree import binomial_price, crr_parameters, implied_volatility


def _bs_call(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    ncdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return S * ncdf(d1) - K * exp(-r * T) * ncdf(d2)


def test_crr_parameters_relationships():
    u, d, p = crr_parameters(0.2, 1.0, 100, r=0.05)
    assert abs(u * d - 1.0) < 1e-12  # d = 1/u by construction
    assert 0.0 <= p <= 1.0


def test_european_call_converges_to_black_scholes():
    bs = _bs_call(100, 100, 1.0, 0.05, 0.20)
    tree = binomial_price(100, 100, 1.0, 0.05, 0.20, n=2000, option="call")
    assert abs(tree - bs) < 0.02


def test_put_call_parity_european():
    # c - p = S - K e^{-rT}
    S, K, T, r, sigma = 100, 95, 1.0, 0.05, 0.25
    c = binomial_price(S, K, T, r, sigma, n=500, option="call")
    p = binomial_price(S, K, T, r, sigma, n=500, option="put")
    assert abs((c - p) - (S - K * exp(-r * T))) < 0.05


def test_american_put_at_least_european():
    euro = binomial_price(100, 110, 1.0, 0.05, 0.30, n=300, option="put")
    amer = binomial_price(100, 110, 1.0, 0.05, 0.30, n=300, option="put", american=True)
    assert amer >= euro - 1e-9


def test_american_call_equals_european_no_dividends():
    # Without dividends, never optimal to exercise an American call early.
    euro = binomial_price(100, 100, 1.0, 0.05, 0.20, n=300, option="call")
    amer = binomial_price(100, 100, 1.0, 0.05, 0.20, n=300, option="call", american=True)
    assert abs(amer - euro) < 1e-6


def test_implied_vol_round_trip():
    price = binomial_price(100, 100, 1.0, 0.05, 0.20, n=300, option="call")
    iv = implied_volatility(price, 100, 100, 1.0, 0.05, option="call", n=300)
    assert abs(iv - 0.20) < 1e-3


def test_invalid_option_type():
    try:
        binomial_price(100, 100, 1.0, 0.05, 0.2, option="banana")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_deep_in_the_money_call_intrinsic():
    # A deep ITM call is worth at least its discounted intrinsic value.
    price = binomial_price(200, 100, 1.0, 0.05, 0.20, n=300, option="call")
    assert price >= 200 - 100 * exp(-0.05)
