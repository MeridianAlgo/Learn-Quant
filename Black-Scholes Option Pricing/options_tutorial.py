"""Interactive Options Pricing Tutorial – Black-Scholes & Greeks.

Run with:
    python options_tutorial.py

Covers the Black-Scholes model, option payoffs, the five Greeks,
put-call parity, and implied volatility. Each section ends with
a short quiz to test understanding.
"""

from __future__ import annotations

import math
from typing import List

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BORDER = "=" * 70


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF using math.erf."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes price for a European call or put.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annual, decimal)
        sigma: Volatility (annual, decimal)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    # put
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Compute the five major Greeks for a call option."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta_call = _norm_cdf(d1)
    delta_put = delta_call - 1.0
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    theta_call = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365.0
    vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0  # per 1% change in vol
    rho_call = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0  # per 1% change in rate
    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "theta_call": theta_call,
        "vega": vega,
        "rho_call": rho_call,
        "d1": d1,
        "d2": d2,
    }


def _ask(question: str, choices: List[str], correct: int, explanation: str) -> None:
    print(f"\n  Q: {question}")
    for i, c in enumerate(choices):
        print(f"     {chr(65 + i)}) {c}")
    while True:
        raw = input("  Your answer (A/B/C/D): ").strip().upper()
        if raw and raw[0] in "ABCD" and ord(raw[0]) - 65 < len(choices):
            break
        print("  Please enter A, B, C, or D.")
    chosen = ord(raw[0]) - 65
    if chosen == correct:
        print("  Correct!")
    else:
        print(f"  Not quite. The answer is {chr(65 + correct)}.")
    print(f"  Explanation: {explanation}\n")


def _header(title: str) -> None:
    print("\n" + BORDER)
    print(title.upper())
    print(BORDER)


# ---------------------------------------------------------------------------
# Section 1 – Option Fundamentals
# ---------------------------------------------------------------------------


def section_fundamentals() -> None:
    _header("Section 1 – Option Fundamentals")

    print(
        """
An OPTION is a contract that gives the buyer the RIGHT (not obligation)
to buy or sell an underlying asset at a specified STRIKE PRICE before
or on an EXPIRY DATE.

TWO BASIC TYPES
  CALL option → right to BUY   (profits when stock rises above strike)
  PUT  option → right to SELL  (profits when stock falls below strike)

KEY TERMS
  S  – Current stock price (underlying)
  K  – Strike price (agreed purchase/sale price)
  T  – Time to expiry (in years)
  r  – Risk-free interest rate (annualised)
  sigma (σ) – Implied volatility of the underlying

PAYOFF AT EXPIRY (not price – ignores premium paid)
  Call payoff = max(S_T - K, 0)
  Put  payoff = max(K - S_T, 0)

MONEYNESS
  In-the-money  (ITM)  Call: S > K        Put: S < K
  At-the-money  (ATM)  Call/Put: S ≈ K
  Out-of-the-money (OTM) Call: S < K      Put: S > K
"""
    )

    S, K = 105.0, 100.0
    call_payoff = max(S - K, 0)
    put_payoff = max(K - S, 0)
    print(f"  Example: Stock = ${S:.2f}, Strike = ${K:.2f}")
    print(f"  Call payoff at expiry = max({S:.2f} - {K:.2f}, 0) = ${call_payoff:.2f}")
    print(f"  Put  payoff at expiry = max({K:.2f} - {S:.2f}, 0) = ${put_payoff:.2f}")
    print(f"  This call is IN-THE-MONEY because S(${S:.2f}) > K(${K:.2f})")

    _ask(
        "A put option has strike K = $50. The stock finishes at $45 at expiry. What is the put's payoff?",
        [
            "$0 — stock is above strike so put expires worthless",
            "$5 — you can sell at $50 when the market price is $45",
            "$45 — the full stock value",
            "$95 — sum of stock and strike",
        ],
        1,
        "Put payoff = max(K - S_T, 0) = max(50 - 45, 0) = $5. "
        "The put is in-the-money because the stock (45) < strike (50).",
    )


# ---------------------------------------------------------------------------
# Section 2 – Black-Scholes Formula
# ---------------------------------------------------------------------------


def section_black_scholes() -> None:
    _header("Section 2 – Black-Scholes Formula")

    print(
        """
The BLACK-SCHOLES MODEL (1973) gives a closed-form price for European options.

ASSUMPTIONS
  - Underlying follows Geometric Brownian Motion (log-normal prices)
  - No dividends
  - Constant volatility and risk-free rate
  - No transaction costs
  - European-style exercise only

CALL PRICE FORMULA
  C = S * N(d1) - K * e^(-rT) * N(d2)

  d1 = [ ln(S/K) + (r + sigma^2/2) * T ] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)

PUT PRICE FORMULA (from put-call parity)
  P = K * e^(-rT) * N(-d2) - S * N(-d1)

where N(x) is the cumulative standard normal distribution.

INTUITION
  N(d2)  = risk-neutral probability the call ends in-the-money
  N(d1)  = delta of the call (see Section 3)
  K*e^(-rT) = present value of the strike price
"""
    )

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    g = greeks(S, K, T, r, sigma)
    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")

    print(f"\n  Inputs: S=${S:.2f}, K=${K:.2f}, T={T:.1f}yr, r={r:.1%}, sigma={sigma:.1%}")
    print(f"  d1 = {g['d1']:.4f}")
    print(f"  d2 = {g['d2']:.4f}")
    print(f"  N(d1) = {_norm_cdf(g['d1']):.4f}")
    print(f"  N(d2) = {_norm_cdf(g['d2']):.4f}")
    print(f"\n  ATM Call price = ${call_price:.4f}")
    print(f"  ATM Put  price = ${put_price:.4f}")

    print(
        """
EFFECT OF INPUTS ON CALL PRICE (all else equal)
  S increases   → Call price increases   (own more intrinsic value)
  K increases   → Call price decreases   (harder to profit)
  T increases   → Call price increases   (more time = more chance)
  r increases   → Call price increases   (PV of strike decreases)
  sigma increases → Call price increases (more uncertainty = more value)
"""
    )

    S_high = 110.0
    call_high = black_scholes(S_high, K, T, r, sigma, "call")
    print(f"  Stock rises from ${S:.2f} to ${S_high:.2f}: Call price ${call_price:.4f} → ${call_high:.4f}")

    _ask(
        "All else equal, what happens to a call option's price when implied volatility INCREASES?",
        [
            "It decreases — higher volatility means more risk so the option is less valuable",
            "It stays the same — volatility does not appear in the payoff formula",
            "It increases — higher volatility increases the probability of large moves",
            "It depends on whether the option is ITM or OTM",
        ],
        2,
        "Higher volatility widens the distribution of possible stock prices at expiry. "
        "Since option payoffs are asymmetric (floored at zero), wider distributions "
        "increase the expected payoff — so the option price rises. "
        "This applies to both calls and puts.",
    )


# ---------------------------------------------------------------------------
# Section 3 – The Greeks
# ---------------------------------------------------------------------------


def section_greeks() -> None:
    _header("Section 3 – The Greeks")

    print(
        """
The GREEKS measure an option's sensitivity to changes in key inputs.
Options traders hedge using the Greeks to manage portfolio risk.

DELTA (delta)
  Change in option price per $1 change in stock price.
  Call delta: 0 to +1  |  Put delta: -1 to 0
  ATM options: delta ≈ ±0.5
  Deep ITM call: delta → +1  (acts like owning the stock)
  Deep OTM call: delta → 0   (barely reacts to stock moves)

GAMMA (gamma)
  Rate of change of delta per $1 change in stock price.
  High gamma = delta changes quickly. Dangerous near expiry for sellers.

THETA (theta)
  Option price decay per day.  Almost always NEGATIVE for option buyers.
  Options lose value every day due to time decay (time value erosion).
  Sellers collect theta; buyers pay it.

VEGA (vega)
  Change in option price per 1% change in implied volatility.
  Both calls and puts have POSITIVE vega.

RHO (rho)
  Change in option price per 1% change in risk-free interest rate.
  Calls have positive rho; puts have negative rho.
"""
    )

    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20  # 3-month ATM
    g = greeks(S, K, T, r, sigma)
    call_price = black_scholes(S, K, T, r, sigma, "call")

    print(f"  ATM 3-month call: S=${S:.2f}, K=${K:.2f}, T={T:.2f}yr, r={r:.1%}, sigma={sigma:.1%}")
    print(f"  Call price = ${call_price:.4f}")
    print(f"\n  Delta (call) = {g['delta_call']:.4f}  (call rises ~${g['delta_call']:.2f} per $1 stock gain)")
    print(f"  Delta (put)  = {g['delta_put']:.4f}  (put falls ~${abs(g['delta_put']):.2f} per $1 stock gain)")
    print(f"  Gamma        = {g['gamma']:.4f}  (delta changes by {g['gamma']:.4f} per $1 stock move)")
    print(f"  Theta (call) = ${g['theta_call']:.4f}/day  (loses this much value per day)")
    print(f"  Vega         = ${g['vega']:.4f} per 1% vol change")
    print(f"  Rho (call)   = ${g['rho_call']:.4f} per 1% rate change")

    print(
        """
DELTA HEDGING
  To make a position delta-neutral, hold -delta shares per option contract.
  Example: You are short 1 call (delta=0.5). Buy 0.5 shares to hedge.
  As the stock moves, delta changes (gamma effect), requiring rebalancing.
"""
    )

    _ask(
        "An option trader is long a call with delta = 0.6. The stock rises $2. "
        "Approximately how much does the call option gain?",
        [
            "$0.30",
            "$1.20",
            "$0.60",
            "$2.00",
        ],
        1,
        "Delta approximates P&L as: gain ≈ delta * price_change = 0.6 * $2 = $1.20. "
        "This is a first-order approximation; the actual gain will differ slightly due to gamma.",
    )

    _ask(
        "An option buyer holds a long call position overnight. All else equal, the option price will:",
        [
            "Increase due to overnight interest accrual",
            "Decrease due to time decay (negative theta)",
            "Stay the same — theta only matters at expiry",
            "Increase if the stock price is above the strike",
        ],
        1,
        "Theta is almost always negative for option buyers. Every day that passes "
        "erodes the time value of the option, reducing its price even if the "
        "stock doesn't move. This is why buying options and sitting on them is costly.",
    )


# ---------------------------------------------------------------------------
# Section 4 – Put-Call Parity
# ---------------------------------------------------------------------------


def section_put_call_parity() -> None:
    _header("Section 4 – Put-Call Parity")

    print(
        """
PUT-CALL PARITY is an arbitrage relationship between call and put prices
for European options on the same underlying with the same strike and expiry.

FORMULA
  C - P = S - K * e^(-rT)

Rearranged:
  C = P + S - K * e^(-rT)   (synthetic call)
  P = C - S + K * e^(-rT)   (synthetic put)

If this relationship breaks, a riskless profit (arbitrage) is possible.

INTUITION: A portfolio of long call + short put has the same payoff as
owning the stock forward at the discounted strike price.
"""
    )

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    C = black_scholes(S, K, T, r, sigma, "call")
    P = black_scholes(S, K, T, r, sigma, "put")
    lhs = C - P
    rhs = S - K * math.exp(-r * T)

    print(f"  S=${S:.2f}, K=${K:.2f}, T={T:.1f}yr, r={r:.1%}, sigma={sigma:.1%}")
    print(f"  Call price C = ${C:.4f}")
    print(f"  Put  price P = ${P:.4f}")
    print(f"\n  LHS: C - P = ${lhs:.4f}")
    print(f"  RHS: S - K*e^(-rT) = ${S:.2f} - ${K * math.exp(-r * T):.4f} = ${rhs:.4f}")
    print(f"  LHS ≈ RHS: {abs(lhs - rhs) < 1e-8}  (verified)")

    print(
        """
APPLICATIONS
  1. Price puts from calls (or vice versa) without re-running the model
  2. Detect mispricings in live options markets
  3. Construct synthetic positions (replicate a call using stock + put)
"""
    )

    _ask(
        "A European call trades at $8, put at $3. Stock = $100, strike = $95, "
        "r = 5%, T = 1 year. K*e^(-rT) = $90.46. "
        "Does put-call parity hold? (C - P = S - K*e^(-rT))",
        [
            "Yes: $8 - $3 = $5, and $100 - $90.46 = $9.54 — parity holds",
            "No: $8 - $3 = $5, but $100 - $90.46 = $9.54 — parity is violated",
            "Cannot check without knowing volatility",
            "Parity only applies to American options",
        ],
        1,
        "LHS = C - P = 8 - 3 = 5. RHS = S - K*e^(-rT) = 100 - 90.46 = 9.54. "
        "5 != 9.54, so parity is violated — an arbitrage opportunity exists.",
    )


# ---------------------------------------------------------------------------
# Section 5 – Implied Volatility
# ---------------------------------------------------------------------------


def section_implied_volatility() -> None:
    _header("Section 5 – Implied Volatility")

    print(
        """
The Black-Scholes model takes volatility as an INPUT and outputs a price.

In practice, option prices are OBSERVED in the market. We can INVERT the
model: given the market price, find the volatility that would produce it.
This is IMPLIED VOLATILITY (IV).

  market_price = BS(S, K, T, r, sigma_IV)  → solve for sigma_IV

Implied volatility is solved numerically (e.g., Newton-Raphson or bisection).

WHY IV MATTERS
  - IV reflects market consensus about future volatility
  - IV > historical volatility → options expensive (sellers have edge)
  - IV < historical volatility → options cheap (buyers have edge)
  - VIX Index = 30-day implied volatility of the S&P 500

VOLATILITY SMILE / SKEW
  In theory, Black-Scholes implies a flat IV across all strikes.
  In practice, IV varies by strike and expiry:
  - Lower strikes (OTM puts) → higher IV (crash insurance premium)
  - This creates a volatility "smile" or "skew" (usually left-skewed for equities)
"""
    )

    # Demonstrate IV by bisection
    target_price = 10.5
    S, K, T, r = 100.0, 100.0, 1.0, 0.05

    lo, hi = 0.001, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        price = black_scholes(S, K, T, r, mid, "call")
        if price < target_price:
            lo = mid
        else:
            hi = mid
    iv = (lo + hi) / 2.0

    print(f"  Market call price = ${target_price:.2f} (S=${S}, K=${K}, T={T}yr, r={r:.1%})")
    print("  Bisection search for implied volatility...")
    print(f"  Implied volatility = {iv:.4f} ({iv * 100:.2f}%)")
    check = black_scholes(S, K, T, r, iv, "call")
    print(f"  Verification: BS({iv:.4f}) = ${check:.4f} ≈ ${target_price:.2f}  (match)")

    print(
        """
VIX AND FEAR
  VIX spikes during market stress (e.g., COVID crash: VIX hit 82.69 in March 2020).
  Traders say: "When the VIX is high, it's time to buy. When VIX is low, look out below."
  High VIX = expensive options (market is pricing in large future moves).
"""
    )

    _ask(
        "A stock's options are priced with IV = 40%. The stock's realised "
        "historical volatility over the past year was 25%. What does this suggest?",
        [
            "Options are cheap — buy them to profit from the volatility gap",
            "Options are expensive — implied > realised, sellers may have an edge",
            "The market expects the stock to fall 40%",
            "IV and historical vol should never differ; this indicates a pricing error",
        ],
        1,
        "When implied volatility (40%) exceeds realised volatility (25%), options "
        "are trading at a premium to what has actually been happening. "
        "Option sellers collect this premium — known as the volatility risk premium. "
        "Buyers overpay on average, though individual outcomes vary.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + BORDER)
    print("OPTIONS PRICING – INTERACTIVE TUTORIAL")
    print(BORDER)
    print(
        """
Welcome! This tutorial covers options pricing from the ground up.
Each section builds on the previous one:

  1. Fundamentals      – Calls, puts, payoffs, moneyness
  2. Black-Scholes     – The formula, inputs, and intuition
  3. The Greeks        – Delta, Gamma, Theta, Vega, Rho
  4. Put-Call Parity   – The arbitrage relationship
  5. Implied Volatility – Inverting the model, the vol smile

Press ENTER to begin each section.
"""
    )

    sections = [
        ("Option Fundamentals", section_fundamentals),
        ("Black-Scholes Formula", section_black_scholes),
        ("The Greeks", section_greeks),
        ("Put-Call Parity", section_put_call_parity),
        ("Implied Volatility", section_implied_volatility),
    ]

    for title, fn in sections:
        input(f"Press ENTER to start: {title} ...")
        fn()

    print("\n" + BORDER)
    print("TUTORIAL COMPLETE")
    print(BORDER)
    print(
        """
Topics covered:
  1. Option Fundamentals   – calls, puts, payoffs, moneyness
  2. Black-Scholes         – formula, d1/d2, log-normal assumption
  3. The Greeks            – delta, gamma, theta, vega, rho
  4. Put-Call Parity       – arbitrage, synthetic positions
  5. Implied Volatility    – inverting BS, vol smile, VIX

Recommended next steps:
  -> UTILS - Finance - Greeks Calculator
  -> UTILS - Advanced Options Pricing
  -> UTILS - Quantitative Methods - Stochastic Processes  (GBM derivation)
"""
    )


if __name__ == "__main__":
    main()
