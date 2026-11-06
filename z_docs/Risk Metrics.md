# Risk Metrics Summary Utility

<p class="lq-meta">Intermediate <span>/</span> Risk & Performance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Risk%20Metrics">source</a></p>


This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

## What Stats Does This Cover?
- **Volatility:** How much returns bounce around (standard deviation)
- **Downside Volatility:** Like volatility, but only counts when the returns fall below zero (focuses on bad swings)
- **Max Drawdown:** The biggest drop from a peak to a low—"worst valley" for your money
- **Skew:** If returns are more to one side (positive for big upswings, negative for big downswings)
- **Kurtosis:** How "chunky" the extremes are (higher means more big outliers)

## Why Bother?
- Professionals use these stats to judge downside risk, stability, and surprise-risk
- "Max drawdown" matters a lot to actual investors (painful losses!)
- Skew and kurtosis give you clues about possible crashes or windfalls

## How to Use
```python
from risk_summary import risk_metrics
import numpy as np
daily_returns = np.random.normal(0.0005, 0.01, 252)
risk_stats = risk_metrics(daily_returns)
for key, value in risk_stats.items():
    print(key, value)
```

## Learn More
- Try this on stocks, crypto, or any investment series
- Combine with Sharpe/Sortino ratios, VaR, and portfolio tools from other UTILS folders for deep analysis

---

## Running it

```bash
cd "Risk Metrics"
python "risk_tutorial.py"
```

## More in Risk & Performance

- [Finance - Calmar Ratio](Finance - Calmar Ratio.md): The Sharpe ratio judges a strategy by how much its returns wobble around their average.
- [Finance - Information Ratio](Finance - Information Ratio.md): When a portfolio is judged against a benchmark, what matters is how much it beat the benchmark by — and how *reliably*.
- [Finance - Jensens Alpha](Finance - Jensens Alpha.md): Beta tells you how much of a portfolio's return came from simply riding the market.
- [Finance - Omega Ratio](Finance - Omega Ratio.md): The Sharpe ratio squeezes a whole return distribution into a mean and a standard deviation, which throws away everything interesting about its shape.
- [Finance - Performance Attribution](Finance - Performance Attribution.md): Brinson decomposition splits portfolio active return into allocation and selection effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*
- [Finance - Treynor Ratio](Finance - Treynor Ratio.md): The Sharpe ratio divides excess return by total volatility, which mixes two very different kinds of risk.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
