"""
Bond Duration, Convexity, and DV01
------------------------------------
Key fixed income analytics for measuring interest rate sensitivity.

- Macaulay Duration: Weighted average time to receive cash flows
- Modified Duration: Price sensitivity to yield changes (% price change per 1% yield change)
- Convexity: Second-order price sensitivity (curvature of price-yield relationship)
- DV01: Dollar value of a basis point (price change for 1bp yield move)
"""

import numpy as np
from typing import Union


def bond_price(cashflows: list, times: list, ytm: float) -> float:
    """
    Calculate bond price as present value of cash flows.

    Args:
        cashflows: List of cash flows (e.g., [coupon, ..., coupon + face]).
        times: List of times in years for each cash flow.
        ytm: Yield to maturity (annual, decimal).

    Returns:
        float: Bond price.
    """
    cashflows = np.array(cashflows)
    times = np.array(times)
    return float(np.sum(cashflows / (1 + ytm) ** times))


def macaulay_duration(cashflows: list, times: list, ytm: float) -> float:
    """
    Macaulay Duration: weighted average time to receive cash flows.

    Returns:
        float: Macaulay duration in years.
    """
    cashflows = np.array(cashflows)
    times = np.array(times)
    pv_flows = cashflows / (1 + ytm) ** times
    price = np.sum(pv_flows)
    return float(np.sum(times * pv_flows) / price)


def modified_duration(cashflows: list, times: list, ytm: float) -> float:
    """
    Modified Duration: % price change per 1% change in yield.

    Returns:
        float: Modified duration.
    """
    mac_dur = macaulay_duration(cashflows, times, ytm)
    return float(mac_dur / (1 + ytm))


def convexity(cashflows: list, times: list, ytm: float) -> float:
    """
    Convexity: second-order price sensitivity to yield changes.

    Returns:
        float: Convexity (in years^2).
    """
    cashflows = np.array(cashflows)
    times = np.array(times)
    price = bond_price(cashflows, times, ytm)
    pv_flows = cashflows / (1 + ytm) ** times
    return float(np.sum(times * (times + 1) * pv_flows) / (price * (1 + ytm) ** 2))


def dv01(cashflows: list, times: list, ytm: float) -> float:
    """
    DV01: Dollar value of a basis point (0.0001 yield move).

    Returns:
        float: Price change for +1bp yield increase.
    """
    price = bond_price(cashflows, times, ytm)
    mod_dur = modified_duration(cashflows, times, ytm)
    return float(price * mod_dur * 0.0001)


def price_change_approx(
    modified_dur: float,
    conv: float,
    price: float,
    yield_change: float,
) -> float:
    """
    Approximate bond price change using duration and convexity.
    dP ≈ -D_mod * P * dy + 0.5 * C * P * dy^2

    Args:
        modified_dur: Modified duration.
        conv: Convexity.
        price: Current bond price.
        yield_change: Change in yield (decimal, e.g., 0.01 for +100bp).

    Returns:
        float: Approximate price change.
    """
    return float(-modified_dur * price * yield_change + 0.5 * conv * price * yield_change**2)


def build_cashflows(
    face: float,
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
) -> tuple:
    """
    Build cash flow schedule for a standard coupon bond.

    Args:
        face: Face (par) value.
        coupon_rate: Annual coupon rate (decimal).
        maturity: Years to maturity.
        frequency: Coupon payments per year (2 = semi-annual).

    Returns:
        tuple: (cashflows list, times list)
    """
    periods = int(maturity * frequency)
    coupon = face * coupon_rate / frequency
    times = [i / frequency for i in range(1, periods + 1)]
    cashflows = [coupon] * periods
    cashflows[-1] += face
    return cashflows, times


if __name__ == "__main__":
    face = 1000
    coupon_rate = 0.05
    maturity = 10
    ytm = 0.04

    cfs, ts = build_cashflows(face, coupon_rate, maturity)
    price = bond_price(cfs, ts, ytm)
    mac_dur = macaulay_duration(cfs, ts, ytm)
    mod_dur = modified_duration(cfs, ts, ytm)
    conv = convexity(cfs, ts, ytm)
    dv = dv01(cfs, ts, ytm)

    print(f"Bond Price:          ${price:.2f}")
    print(f"Macaulay Duration:   {mac_dur:.4f} years")
    print(f"Modified Duration:   {mod_dur:.4f}")
    print(f"Convexity:           {conv:.4f}")
    print(f"DV01:                ${dv:.4f}")

    dp = price_change_approx(mod_dur, conv, price, 0.01)
    print(f"\nApprox price change (+100bp): ${dp:.2f}")
