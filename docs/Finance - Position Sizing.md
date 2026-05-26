# Finance – Position Sizing

## Overview

**Position sizing is the most underrated skill in quantitative trading.** A strategy with a mediocre edge and excellent position sizing will outperform a brilliant strategy with reckless sizing. This module covers four fundamental frameworks every trader and quant must understand before risking real capital.

## Key Concepts

### **Why Position Sizing Matters**
- Two traders with the *same strategy* and the *same edge* can have dramatically different outcomes based solely on how much they bet per trade.
- Over-betting leads to catastrophic drawdowns even with a positive expected value.
- Under-betting leaves profits on the table and may not cover transaction costs.

### **1. Fixed Fractional**
Risk a constant percentage of your portfolio on every trade:

```
dollar_risk = portfolio × risk_pct
position_size = dollar_risk / stop_loss_pct
```

Example: Risk 1% of $100,000 with a 5% stop → buy $20,000 of stock.

**Pros:** Simple, scales with portfolio, well-understood.
**Cons:** Doesn't adapt to strategy's actual edge or volatility conditions.

### **2. Kelly Criterion**
The mathematically optimal bet fraction for maximum long-run compound growth:

```
f* = p – q/b = p – (1 – p) / (avg_win / avg_loss)
```

| Term | Meaning |
|------|---------|
| p | Win probability |
| q = 1-p | Loss probability |
| b | Net odds (avg win / avg loss) |

> **Practical rule**: Always use Half-Kelly (`f*/2`) or less. Full Kelly produces extreme drawdowns that most traders cannot tolerate psychologically.

### **3. Volatility Targeting**
Scale positions so the portfolio hits a constant target volatility:

```
notional = portfolio × (target_vol / asset_vol)
```

When a stock's volatility doubles, you halve your position size — keeping dollar risk constant. Used by Risk Parity funds and Managed Futures CTAs.

### **4. Risk of Ruin**
The probability of losing enough capital to be unable to continue trading:

```
Risk of Ruin ≈ ((1 – edge) / (1 + edge))^(capital / risk_per_trade)
```

In practice, estimated via Monte Carlo over thousands of simulated trading careers.

## Files
- `position_sizing_tutorial.py`: Fixed fractional calculator, Kelly criterion with growth simulation, volatility targeting, and Monte Carlo Risk of Ruin.

## How to Run
```bash
python position_sizing_tutorial.py
```

## Financial Applications

### 1. Discretionary Trading
- Fixed fractional (1–2% risk per trade) is the standard rule taught in all professional trading courses.
- Most prop firms enforce maximum risk-per-trade rules contractually.

### 2. Systematic / Algorithmic Trading
- Kelly is used to size signals in multi-strategy systems (allocate more Kelly-fraction to higher-edge strategies).
- Volatility targeting is the default in Commodity Trading Advisors (CTAs) for futures positions.

### 3. Options Trading
- Greeks-based sizing: position size chosen to limit delta exposure to 1% of portfolio.
- Theta decay strategies (selling options) often use Kelly-like sizing based on edge estimates.

### 4. Portfolio Construction
- Risk Parity: every asset contributes equally to portfolio volatility via inverse-vol weighting.
- Maximum Sharpe portfolios from mean-variance optimisation often implicitly implement Kelly logic.

## Best Practices

- **Never use Full Kelly in practice**: estimation error in win_prob and win_loss_ratio is significant, and Kelly's variance is unbounded near the optimum.
- **Risk of Ruin > 5%? Don't trade**: any strategy with meaningful ruin probability should be either improved or sized down.
- **Re-calculate volatility targets frequently**: asset volatility changes — update positions at least monthly (daily for liquid futures).
- **Account for correlation**: if trading multiple strategies, their combined Kelly fraction depends on their correlation structure (use portfolio-level Kelly).
- **Transaction costs**: include slippage and commissions when estimating win_prob and win_loss_ratio — overestimating edge is the #1 cause of over-sizing.