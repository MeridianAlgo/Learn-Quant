"""Position Sizing Strategies for Quantitative Trading.

Run with:
    python position_sizing_tutorial.py

Even the best trading strategy can lose money — or destroy a portfolio — with
poor position sizing.  This tutorial covers four fundamental frameworks that
every quantitative trader needs to understand:

1. Fixed Fractional:     Risk a constant % of portfolio per trade.
2. Kelly Criterion:      Mathematically optimal bet size to maximise long-run growth.
3. Volatility Targeting: Size positions so the portfolio targets a constant volatility.
4. Risk of Ruin:         Probability of catastrophic capital loss (via Monte Carlo).

Key Concepts:
- Edge (Expected Value):  Average profit per unit risked, calculated over many trades.
- Kelly Fraction (f*):    f* = p – q/b  (win_prob – loss_prob / odds).
- Drawdown:               Peak-to-trough decline in portfolio equity.
- Volatility Targeting:   position_notional = portfolio * (target_vol / asset_vol).
- Risk of Ruin:           Probability of equity falling below a "ruin" threshold.
"""

import numpy as np
import pandas as pd

# ─── 1. FIXED FRACTIONAL POSITION SIZING ────────────────────────────────────


def fixed_fractional_sizing(portfolio_value: float, risk_per_trade_pct: float, stop_loss_pct: float) -> dict:
    """
    Calculate position size using the Fixed Fractional method.

    This is the most widely used position sizing technique in retail and
    professional trading.  The logic:
    1. Decide how much of your portfolio you will risk per trade (e.g., 1%).
    2. Set a stop-loss at a fixed % below your entry (e.g., 5%).
    3. Back-calculate the position size so that if the stop is hit, you lose exactly $risk.

    Example walk-through:
        Portfolio:            $100,000
        Risk per trade (1%):  $1,000 maximum loss
        Stop-loss:            5% below entry
        Maximum position:     $1,000 / 5% = $20,000

    Advantages:
    - Scales naturally as the portfolio grows or shrinks.
    - Never risks more than the chosen fraction in one trade.
    - Easy to implement and audit.

    Args:
        portfolio_value:    Current total portfolio value in dollars.
        risk_per_trade_pct: Fraction of portfolio to risk (e.g., 0.01 for 1%).
        stop_loss_pct:      Stop-loss distance as a fraction of entry price (e.g., 0.05 for 5%).

    Returns:
        Dict with dollar_risk, position_value, and position as % of portfolio.
    """
    # Dollar amount you are willing to lose if the stop-loss fires
    dollar_risk = portfolio_value * risk_per_trade_pct

    # If the stock drops stop_loss_pct before you exit, you lose dollar_risk.
    # So: position_value × stop_loss_pct = dollar_risk
    # → position_value = dollar_risk / stop_loss_pct
    position_value = dollar_risk / stop_loss_pct

    return {
        "portfolio_value": portfolio_value,
        "dollar_risk": dollar_risk,
        "position_value": position_value,
        "position_pct_of_portfolio": position_value / portfolio_value,
    }


# ─── 2. KELLY CRITERION ──────────────────────────────────────────────────────


def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
    """
    Calculate the Kelly optimal fraction of capital to bet per trade.

    The Kelly Criterion (John Kelly, 1956) finds the bet size that maximises
    the *expected logarithm* of wealth — which is equivalent to maximising the
    long-run compound growth rate.

    For a binary bet (win or lose) the formula is:
        f* = p – q/b  =  p – (1–p)/b

    where:
        p = probability of winning
        q = 1 – p = probability of losing
        b = net odds (profit per dollar risked; e.g., b=2 → win $2 for $1 bet)

    Key properties:
    - f* > 0 ↔ the strategy has a positive expected value (edge).
    - f* < 0 ↔ the strategy has a negative expected value — do not trade!
    - f* = 1 ↔ bet everything each trade (only valid when you never lose).
    - Practical usage: most traders use Half-Kelly (f*/2) to reduce variance.

    Why not always bet Kelly?
    - The optimal Kelly path is extremely volatile even for profitable strategies.
    - Estimation error: real win_prob and win_loss_ratio are uncertain estimates.
    - Half-Kelly gives ~75% of the growth rate with ~50% of the variance.

    Args:
        win_prob:       Historical probability of a winning trade (e.g., 0.55 for 55%).
        win_loss_ratio: Average win size / average loss size (e.g., 1.5 means avg win = 1.5x avg loss).

    Returns:
        Kelly fraction f* ∈ [0, 1].  Returns 0 if strategy has no edge.
    """
    p = win_prob
    q = 1.0 - p
    b = win_loss_ratio  # net odds: profit per dollar of risk

    kelly_f = p - (q / b)

    # A negative Kelly means no edge — return 0 to avoid betting in a losing game
    return max(0.0, kelly_f)


def kelly_growth_simulation(
    kelly_f: float, win_prob: float, win_loss_ratio: float, n_trades: int = 500, seed: int = 42
) -> pd.DataFrame:
    """
    Simulate portfolio growth under Full Kelly, Half Kelly, Quarter Kelly, and Fixed 1%.

    This visualises the famous Kelly trade-off:
    - Full Kelly maximises growth but creates extreme drawdowns along the way.
    - Half Kelly sacrifices some growth for dramatically lower variance.
    - Quarter Kelly is conservative but still grows faster than Fixed 1% if edge is large.

    All portfolios are normalised to start at $1.

    Args:
        kelly_f:        Full Kelly fraction (output of kelly_criterion()).
        win_prob:       Probability of winning each trade.
        win_loss_ratio: Average win / average loss ratio.
        n_trades:       Number of sequential trades to simulate.
        seed:           Random seed.

    Returns:
        DataFrame with columns: Trade (0 to n_trades), Full Kelly, Half Kelly,
        Quarter Kelly, Fixed 1%.
    """
    rng = np.random.default_rng(seed)

    # Simulate a sequence of win/loss outcomes
    outcomes = rng.random(n_trades) < win_prob  # True = win

    fractions = {
        "Full Kelly": kelly_f,
        "Half Kelly": kelly_f / 2,
        "Quarter Kelly": kelly_f / 4,
        "Fixed 1%": 0.01,
    }

    results: dict = {"Trade": list(range(n_trades + 1))}

    for name, f in fractions.items():
        portfolio = [1.0]  # start at $1 (normalised)
        for win in outcomes:
            current = portfolio[-1]
            if win:
                # Win: portfolio grows proportionally to the fraction and odds
                portfolio.append(current * (1.0 + f * win_loss_ratio))
            else:
                # Loss: portfolio shrinks by the bet fraction
                portfolio.append(current * (1.0 - f))
        results[name] = portfolio

    return pd.DataFrame(results)


# ─── 3. VOLATILITY TARGETING ─────────────────────────────────────────────────


def volatility_targeting(
    target_annual_vol: float,
    asset_annual_vol: float,
    portfolio_value: float,
    asset_price: float,
    shares_per_lot: int = 1,
) -> dict:
    """
    Calculate the position size needed to achieve a target portfolio volatility.

    Volatility targeting maintains *constant risk* as market volatility changes.
    - In low-volatility environments → size up (larger positions).
    - In high-volatility environments → size down (smaller positions).

    This is the core of many Risk Parity strategies and Managed Futures funds.
    The formula simply scales exposure inversely with asset volatility:

        notional_exposure = portfolio_value × (target_vol / asset_vol)

    Intuition: if an asset is twice as volatile as our target allows, we hold
    half the notional we otherwise would — keeping dollar volatility constant.

    Args:
        target_annual_vol: Target annual portfolio volatility (e.g., 0.10 for 10%).
        asset_annual_vol:  Current annual volatility of the asset (estimate from recent returns).
        portfolio_value:   Total portfolio size in dollars.
        asset_price:       Current price per share/contract.
        shares_per_lot:    Minimum tradeable lot size (default 1 share = standard equities).

    Returns:
        Dict with notional_exposure, n_shares, and actual_portfolio_vol.
    """
    # Compute the notional value of the position that will deliver the target vol
    notional_exposure = portfolio_value * (target_annual_vol / asset_annual_vol)

    # Convert to integer shares (round down to nearest lot to avoid over-exposure)
    n_shares = int(notional_exposure / (asset_price * shares_per_lot)) * shares_per_lot

    # Actual portfolio vol after rounding to whole shares
    actual_notional = n_shares * asset_price
    actual_vol = asset_annual_vol * (actual_notional / portfolio_value)

    return {
        "target_vol": target_annual_vol,
        "asset_vol": asset_annual_vol,
        "notional_exposure": notional_exposure,
        "n_shares": n_shares,
        "actual_portfolio_vol": actual_vol,
    }


# ─── 4. RISK OF RUIN ────────────────────────────────────────────────────────


def risk_of_ruin(
    win_prob: float,
    win_loss_ratio: float,
    risk_per_trade_pct: float,
    n_simulations: int = 10000,
    ruin_threshold: float = 0.50,
    n_trades: int = 1000,
    seed: int = 42,
) -> float:
    """
    Estimate the Risk of Ruin via Monte Carlo simulation.

    "Risk of Ruin" is the probability that a trader loses enough capital to be
    effectively "ruined" — either unable to continue trading or suffering a loss
    large enough to be psychologically or financially devastating.

    We define ruin conservatively: portfolio value falls below `ruin_threshold`
    of the starting capital (e.g., losing more than 50%).

    Why Monte Carlo?
    - The analytical formula assumes binary outcomes (win fixed amount / lose fixed amount).
    - Real trading has variable win/loss sizes.  Monte Carlo handles any distribution.
    - Monte Carlo is also more intuitive: just run thousands of "trader careers" and count how many go bust.

    Args:
        win_prob:           Historical win rate of the strategy (e.g., 0.55 = 55% wins).
        win_loss_ratio:     Average win / average loss size.
        risk_per_trade_pct: Fraction of *current* portfolio risked per trade (fixed fractional).
        n_simulations:      Number of independent Monte Carlo paths (more = more accurate estimate).
        ruin_threshold:     Portfolio value at which ruin is declared (e.g., 0.50 = 50% of start).
        n_trades:           Number of trades per simulated "career".
        seed:               Random seed.

    Returns:
        Estimated probability of ruin (float in [0, 1]).
    """
    rng = np.random.default_rng(seed)
    ruin_count = 0

    for _ in range(n_simulations):
        portfolio = 1.0  # start at $1 (normalised)

        # Simulate a full trading career of n_trades
        outcomes = rng.random(n_trades) < win_prob

        for win in outcomes:
            if win:
                # Winning trade: grow by risk_pct × win_loss_ratio
                portfolio *= 1.0 + risk_per_trade_pct * win_loss_ratio
            else:
                # Losing trade: shrink by risk_pct
                portfolio *= 1.0 - risk_per_trade_pct

            # Check for ruin at each step (stop simulating once ruined)
            if portfolio < ruin_threshold:
                ruin_count += 1
                break  # this path is ruined; start next simulation

    return ruin_count / n_simulations


# ─── MAIN ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("POSITION SIZING STRATEGIES")
    print("=" * 60)

    # ── 1. Fixed Fractional ──
    print("\n[1] Fixed Fractional Position Sizing")
    print("  Rule: risk exactly 1% of portfolio per trade with a 5% stop-loss.")
    ff = fixed_fractional_sizing(portfolio_value=100_000, risk_per_trade_pct=0.01, stop_loss_pct=0.05)
    print(f"  Portfolio value:         ${ff['portfolio_value']:>10,.0f}")
    print(f"  Dollar risk (1%):        ${ff['dollar_risk']:>10,.0f}")
    print(
        f"  Position size:           ${ff['position_value']:>10,.0f}  "
        f"({ff['position_pct_of_portfolio']:.0%} of portfolio)"
    )
    print("  Interpretation: buy $20,000 of stock with a $1,000 max loss.")

    # ── 2. Kelly Criterion ──
    print("\n[2] Kelly Criterion")
    win_p = 0.55  # 55% of trades are winners
    wl_ratio = 1.5  # average winner is 1.5× the average loser
    kelly_f = kelly_criterion(win_p, wl_ratio)
    half_kelly = kelly_f / 2

    print(f"  Win probability:          {win_p:.0%}")
    print(f"  Win/Loss ratio:           {wl_ratio:.1f}x")
    print(f"  Expected value per trade: {win_p * wl_ratio - (1 - win_p):.4f}")
    print(f"  Full Kelly fraction:      {kelly_f:.2%}  per trade")
    print(f"  Half Kelly (recommended): {half_kelly:.2%}  per trade")

    print("\n  Simulating 500 trades starting with $1 (normalised portfolio):")
    growth = kelly_growth_simulation(kelly_f, win_p, wl_ratio, n_trades=500)
    final = growth.iloc[-1]
    for col in ["Full Kelly", "Half Kelly", "Quarter Kelly", "Fixed 1%"]:
        print(f"    {col:<18}: ${final[col]:.2f}")
    print("  Note: Full Kelly grows fastest but suffers the largest drawdowns along the way.")

    # ── 3. Volatility Targeting ──
    print("\n[3] Volatility Targeting")
    print("  Goal: position the portfolio so annual volatility = 10%, regardless of asset vol.")
    vt = volatility_targeting(target_annual_vol=0.10, asset_annual_vol=0.25, portfolio_value=500_000, asset_price=150.0)
    print(f"  Portfolio:                ${vt['portfolio_value']:>10,.0f}")
    print(f"  Asset annual volatility:   {vt['asset_vol']:.0%}")
    print(f"  Target portfolio vol:      {vt['target_vol']:.0%}")
    print(
        f"  Required notional:        ${vt['notional_exposure']:>10,.0f}  "
        f"({vt['notional_exposure'] / vt['portfolio_value']:.0%} of portfolio)"
    )
    print(f"  Shares to buy:             {vt['n_shares']:,}")
    print(f"  Actual portfolio vol:      {vt['actual_portfolio_vol']:.2%}  (after rounding)")
    print("  Key insight: asset vol is 25% but we target 10%, so we only invest 40% of portfolio.")

    # ── 4. Risk of Ruin ──
    print("\n[4] Risk of Ruin (Monte Carlo: 5,000 paths × 1,000 trades)")
    print("  Ruin = losing more than 50% of starting capital.\n")

    scenarios = [
        ("Aggressive  (5% risk, edge=+55%/1.5x)", 0.55, 1.5, 0.05),
        ("Moderate    (2% risk, edge=+55%/1.5x)", 0.55, 1.5, 0.02),
        ("Conservative(1% risk, edge=+55%/1.5x)", 0.55, 1.5, 0.01),
        ("Poor edge   (1% risk, edge=+48%/1.2x)", 0.48, 1.2, 0.01),
    ]

    results_data = []
    for label, wp, wlr, rpt in scenarios:
        ror = risk_of_ruin(wp, wlr, rpt, n_simulations=5000, n_trades=1000)
        results_data.append({"Scenario": label, "Risk of Ruin": f"{ror:.1%}"})
        print(f"  {label}: {ror:.1%}")

    print()
    print("  Key insight: aggressive sizing (5% per trade) creates significant ruin risk")
    print("  even with a genuine edge.  Poor edge strategies are dangerous at any size.")

    print("\nPosition Sizing tutorial complete!")
    print("Lesson: A profitable strategy + wrong position sizing = losing portfolio.")
    print("        Always calculate Kelly, always use Half-Kelly or less in practice.")


if __name__ == "__main__":
    main()
