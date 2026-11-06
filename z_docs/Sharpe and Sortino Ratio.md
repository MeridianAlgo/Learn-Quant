# Sharpe and Sortino Ratio Calculator

<p class="lq-meta">Intermediate <span>/</span> Risk & Performance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Sharpe%20and%20Sortino%20Ratio">source</a></p>


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

## Running it

```bash
cd "Sharpe and Sortino Ratio"
python "ratio_calculator.py"
```

## More in Risk & Performance

- [Finance - Calmar Ratio](Finance - Calmar Ratio.md): The Sharpe ratio judges a strategy by how much its returns wobble around their average.
- [Finance - Information Ratio](Finance - Information Ratio.md): When a portfolio is judged against a benchmark, what matters is how much it beat the benchmark by — and how *reliably*.
- [Finance - Jensens Alpha](Finance - Jensens Alpha.md): Beta tells you how much of a portfolio's return came from simply riding the market.
- [Finance - Omega Ratio](Finance - Omega Ratio.md): The Sharpe ratio squeezes a whole return distribution into a mean and a standard deviation, which throws away everything interesting about its shape.
- [Finance - Performance Attribution](Finance - Performance Attribution.md): Brinson decomposition splits portfolio active return into allocation and selection effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*
- [Finance - Treynor Ratio](Finance - Treynor Ratio.md): The Sharpe ratio divides excess return by total volatility, which mixes two very different kinds of risk.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
