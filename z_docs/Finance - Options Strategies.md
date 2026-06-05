<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Options Strategies"
    python "options_strategies.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Options%20Strategies)

---
# Finance – Options Strategies

## Overview

Options strategies combine multiple option legs (calls and puts at different strikes and expiries) to create specific risk/reward profiles. Rather than taking a directional bet with a single option, multi-leg strategies allow traders to express nuanced views on direction, volatility, time decay, and risk limits.

This module implements the most widely used options strategies with payoff calculations, breakeven analysis, and maximum profit/loss profiles.

## Key Concepts

### Building Blocks

| Leg | Description | Max Profit | Max Loss |
|-----|-------------|-----------|---------|
| Long Call | Right to buy at strike K | Unlimited | Premium paid |
| Short Call | Obligation to sell at strike K | Premium received | Unlimited |
| Long Put | Right to sell at strike K | K - Premium | Premium paid |
| Short Put | Obligation to buy at strike K | Premium received | K - Premium |

### Strategy Classification

**Directional strategies**: profit from price moving in a specific direction with limited risk.
- Bull Call Spread, Bear Put Spread, Risk Reversal.

**Volatility strategies**: profit from large moves (long vol) or small moves (short vol).
- Long Straddle, Long Strangle, Iron Condor, Iron Butterfly.

**Income strategies**: collect premium in exchange for taking on tail risk.
- Covered Call, Cash-Secured Put, Iron Condor.

### Key Metrics for Every Strategy

- **Maximum Profit**: the most the strategy can earn at expiry.
- **Maximum Loss**: the most the strategy can lose at expiry (should always be defined before entering).
- **Breakeven(s)**: underlying price(s) at which the strategy breaks even at expiry.
- **Net Premium**: total premium paid (debit) or received (credit) for all legs combined.

### Spread vs Naked

Spreads cap both profit and loss, reducing margin requirements and risk significantly compared to naked (single-leg) positions. Professional traders almost always use spreads rather than naked shorts.

## Files
- `options_strategies.py`: Bull call spread, bear put spread, long straddle, long strangle, iron condor, iron butterfly, covered call, and protective put — each with P&L profile and breakeven calculations.

## How to Run
```bash
python options_strategies.py
```

## Strategies Covered

### Bull Call Spread
- **View**: moderately bullish.
- **Structure**: long call at K1, short call at K2 (K2 > K1), same expiry.
- **Max Profit**: (K2 - K1) - net_debit, achieved when underlying >= K2.
- **Max Loss**: net_debit, when underlying <= K1.
- **Breakeven**: K1 + net_debit.

### Iron Condor
- **View**: low volatility, underlying stays in a range.
- **Structure**: short put at K1, long put at K2, short call at K3, long call at K4 (K2<K1<K3<K4).
- **Max Profit**: net_credit received, when K1 <= underlying <= K3.
- **Max Loss**: (K1 - K2) - net_credit or (K4 - K3) - net_credit.
- **Breakeven**: K1 - net_credit (lower) and K3 + net_credit (upper).

### Long Straddle
- **View**: high volatility, large move in either direction.
- **Structure**: long call + long put at the same strike and expiry.
- **Max Profit**: unlimited (call side) or strike - premium (put side).
- **Max Loss**: total premium paid, when underlying = strike at expiry.

## Financial Applications

### 1. Earnings Plays
- Long straddle or strangle before earnings: profit if the stock moves more than the implied move priced into the options.
- Iron condor after earnings: collect premium if the post-earnings vol crush is larger than the residual move.

### 2. Defined-Risk Speculation
- Bull call spreads allow directional bets with a fixed maximum loss — ideal for portfolios with strict drawdown limits.

### 3. Income Generation
- Iron condors on indices (SPX, NDX) generate steady premium in low-volatility environments.
- Covered calls on existing equity positions generate income while capping upside.

### 4. Volatility Arbitrage
- If implied volatility is expensive relative to expected realised vol, sell vega via iron condors or strangles.
- If implied vol is cheap, buy straddles or strangles.

## Best Practices

- **Always know your max loss before entering**: Never enter a strategy where the maximum loss is undefined or unlimited unless specifically authorised.
- **Consider transaction costs**: Multi-leg strategies incur commissions on each leg — net premium must be large enough to cover costs.
- **Monitor gamma near expiry**: Short gamma positions (iron condor, short straddle) become increasingly dangerous as expiry approaches; the P&L can move violently on small underlying moves.
- **Roll or close early**: Most income strategies should be closed at 50% of max profit rather than held to expiry — this reduces gamma risk and frees capital for new trades.


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
