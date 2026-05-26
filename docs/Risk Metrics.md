# Risk Metrics Summary Utility

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
