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
