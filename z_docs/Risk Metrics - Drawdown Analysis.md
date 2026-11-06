# Drawdown Analysis

<p class="lq-meta">Intermediate <span>/</span> Risk & Performance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Risk%20Metrics%20-%20Drawdown%20Analysis">source</a></p>


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

## Running it

```bash
cd "Risk Metrics - Drawdown Analysis"
python "drawdown_analysis.py"
```

## More in Risk & Performance

- [Finance - Calmar Ratio](Finance - Calmar Ratio.md): The Sharpe ratio judges a strategy by how much its returns wobble around their average.
- [Finance - Information Ratio](Finance - Information Ratio.md): When a portfolio is judged against a benchmark, what matters is how much it beat the benchmark by — and how *reliably*.
- [Finance - Jensens Alpha](Finance - Jensens Alpha.md): Beta tells you how much of a portfolio's return came from simply riding the market.
- [Finance - Omega Ratio](Finance - Omega Ratio.md): The Sharpe ratio squeezes a whole return distribution into a mean and a standard deviation, which throws away everything interesting about its shape.
- [Finance - Performance Attribution](Finance - Performance Attribution.md): Brinson decomposition splits portfolio active return into allocation and selection effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*
- [Finance - Treynor Ratio](Finance - Treynor Ratio.md): The Sharpe ratio divides excess return by total volatility, which mixes two very different kinds of risk.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
