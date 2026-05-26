import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Duration Convexity"))
from duration_convexity import (
    bond_price,
    build_cashflows,
    convexity,
    dv01,
    macaulay_duration,
    modified_duration,
    price_change_approx,
)


@pytest.fixture
def par_bond():
    """5% coupon, 10yr, priced at par (ytm=5%)."""
    cfs, ts = build_cashflows(1000, 0.05, 10, frequency=2)
    return cfs, ts


def test_bond_price_reasonable(par_bond):
    """5% coupon bond at 5% yield should trade near par (small deviation due to annual-effective discounting)."""
    cfs, ts = par_bond
    price = bond_price(cfs, ts, 0.05)
    assert 990 < price < 1010


def test_bond_price_premium_at_lower_yield(par_bond):
    cfs, ts = par_bond
    price = bond_price(cfs, ts, 0.04)
    assert price > 1000


def test_bond_price_discount_at_higher_yield(par_bond):
    cfs, ts = par_bond
    price = bond_price(cfs, ts, 0.06)
    assert price < 1000


def test_macaulay_duration_positive(par_bond):
    cfs, ts = par_bond
    dur = macaulay_duration(cfs, ts, 0.05)
    assert dur > 0
    assert dur < 10  # Less than maturity for coupon bond


def test_modified_less_than_macaulay(par_bond):
    cfs, ts = par_bond
    ytm = 0.05
    mac = macaulay_duration(cfs, ts, ytm)
    mod = modified_duration(cfs, ts, ytm)
    assert mod < mac
    assert abs(mod - mac / (1 + ytm)) < 1e-6  # annual-effective: divide by (1 + ytm)


def test_convexity_positive(par_bond):
    cfs, ts = par_bond
    conv = convexity(cfs, ts, 0.05)
    assert conv > 0


def test_dv01_small_positive(par_bond):
    cfs, ts = par_bond
    dv = dv01(cfs, ts, 0.05)
    assert dv > 0
    assert dv < 1  # Should be fraction of a dollar per $1000 bond


def test_price_change_approx_negative_for_yield_rise(par_bond):
    cfs, ts = par_bond
    mod_dur = modified_duration(cfs, ts, 0.05)
    conv = convexity(cfs, ts, 0.05)
    price = bond_price(cfs, ts, 0.05)
    dp = price_change_approx(mod_dur, conv, price, 0.01)
    assert dp < 0  # Price falls when yield rises


def test_build_cashflows_length():
    cfs, ts = build_cashflows(1000, 0.05, 10, frequency=2)
    assert len(cfs) == 20
    assert len(ts) == 20
    assert cfs[-1] > cfs[0]  # Last payment includes face value
