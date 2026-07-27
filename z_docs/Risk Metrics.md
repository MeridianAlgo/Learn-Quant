<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Risk Metrics"
    python "risk_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Risk%20Metrics)

---
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


---

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Calmar Ratio](Finance - Calmar Ratio.md)__

    The Sharpe ratio judges a strategy by how much its returns wobble around their

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Jensens Alpha](Finance - Jensens Alpha.md)__

    Beta tells you how much of a portfolio's return came from simply riding the

-   :material-shield-alert-outline: __[Finance - Omega Ratio](Finance - Omega Ratio.md)__

    The Sharpe ratio squeezes a whole return distribution into a mean and a

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Finance - Treynor Ratio](Finance - Treynor Ratio.md)__

    The Sharpe ratio divides excess return by total volatility, which mixes two

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
