<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Sharpe and Sortino Ratio"
    python "ratio_calculator.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Sharpe%20and%20Sortino%20Ratio)

---
# Sharpe and Sortino Ratio Calculator

This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

## What are the Sharpe and Sortino Ratios?
**Sharpe Ratio** measures how much excess return you get for each unit of total risk you take (as measured by volatility).
**Sortino Ratio** is similar, but only counts downside risk, ignoring upside swings.

- **Higher values** mean better risk-adjusted returns.
- Used by professional and retail investors to evaluate stock, fund, and portfolio performance.

## How to Use
1. Make sure you have Python and `numpy` installed.
2. Put your list or array of returns (like daily or monthly returns) in the code.
3. Call the `sharpe_ratio()` or `sortino_ratio()` function from the `ratio_calculator.py` file.
4. Optionally, set the risk-free rate and number of periods per year.

### Example
```python
from ratio_calculator import sharpe_ratio, sortino_ratio
import numpy as np

daily_returns = np.random.normal(0.001, 0.01, 252)
print("Sharpe Ratio:", sharpe_ratio(daily_returns))
print("Sortino Ratio:", sortino_ratio(daily_returns))
```

## Why Does This Matter?
- Compare the quality of investments, not just raw returns.
- Identify if an asset rewards you for the risk you take.
- Learn core principles of risk management and portfolio analysis.

*For more finance learning, check /Documentation or see other UTILS modules!*


---

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Calmar Ratio](Finance - Calmar Ratio.md)__

    The Sharpe ratio judges a strategy by how much its returns wobble around their

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

-   :material-shield-alert-outline: __[Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md)__

    Comprehensive drawdown metrics for quantifying portfolio loss risk over time. Drawdown measures capture both the **depth** and **duration** of losses — dimensions VaR ignores.

-   :material-shield-alert-outline: __[Risk Metrics - Stress Testing](Risk Metrics - Stress Testing.md)__

    Stress tests answer: *"What happens if 2008 repeats?"* or *"How big a shock kills the portfolio?"* Required by Basel III, CCAR, and most institutional risk frameworks.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
