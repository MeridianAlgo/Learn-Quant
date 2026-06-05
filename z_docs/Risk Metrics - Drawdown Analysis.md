<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Risk Metrics - Drawdown Analysis"
    python "drawdown_analysis.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Risk%20Metrics%20-%20Drawdown%20Analysis)

---
# Drawdown Analysis

Comprehensive drawdown metrics for quantifying portfolio loss risk over time. Drawdown measures capture both the **depth** and **duration** of losses — dimensions VaR ignores.

## Functions

| Function | Description |
|---|---|
| `drawdown_series(returns)` | Drawdown at each point in time |
| `max_drawdown(returns)` | Largest peak-to-trough decline |
| `calmar_ratio(returns, periods)` | Annualized return / Max drawdown |
| `ulcer_index(returns)` | RMS of drawdown depths |
| `ulcer_performance_index(returns, rf)` | Mean excess return / Ulcer index |
| `average_drawdown(returns)` | Mean depth across all drawdown periods |
| `max_drawdown_duration(returns)` | Longest continuous drawdown in periods |
| `drawdown_summary(returns, periods)` | All metrics in one dict |

## Key Concepts

- **Max Drawdown**: The worst loss from a peak. MDD of 0.25 = portfolio dropped 25% from its high before recovering.
- **Calmar Ratio**: Return per unit of drawdown risk. Like Sharpe but uses MDD instead of std dev. Higher is better.
- **Ulcer Index**: Named for the "ulcer-inducing" anxiety of prolonged losses. RMS penalizes long drawdowns heavily.
- **UPI (Martin Ratio)**: Return / Ulcer Index. Better than Calmar for comparing strategies with similar MDD but different recovery times.

## Example

```python
from drawdown_analysis import drawdown_summary
import numpy as np

returns = np.random.normal(0.0005, 0.015, 504)
summary = drawdown_summary(returns)
# {'max_drawdown': 0.142, 'calmar_ratio': 0.87, 'ulcer_index': 0.032, ...}
```

## Benchmarks

| Strategy | Typical Max Drawdown |
|---|---|
| Long-only equity | 30–60% |
| 60/40 portfolio | 20–35% |
| Market-neutral HF | 5–15% |
| Trend following | 15–30% |


---

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

-   :material-shield-alert-outline: __[Risk Metrics - Stress Testing](Risk Metrics - Stress Testing.md)__

    Stress tests answer: *"What happens if 2008 repeats?"* or *"How big a shock kills the portfolio?"* Required by Basel III, CCAR, and most institutional risk frameworks.

-   :material-shield-alert-outline: __[Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md)__

    This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

-   :material-shield-alert-outline: __[Value at Risk (VaR)](Value at Risk (VaR).md)__

    **Value at Risk** is the single most widely quoted number in financial risk

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
