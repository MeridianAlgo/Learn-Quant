# Correlation Analysis

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Correlation%20Analysis">source</a></p>


Analyze correlations between financial instruments for portfolio construction and risk management.

## Features

- Pearson Correlation
- Rolling Correlation
- Correlation Matrix
- EWMA Correlation
- Rank Correlation (Spearman)
- Tail Correlation
- Correlation Stability Analysis

## Usage

```python
from correlation_analysis import pearson_correlation, rolling_correlation

returns1 = [0.01, -0.02, 0.015, -0.01, 0.02]
returns2 = [0.008, -0.015, 0.012, -0.008, 0.018]

corr = pearson_correlation(returns1, returns2)
print(f"Correlation: {corr:.4f}")

rolling = rolling_correlation(returns1, returns2, window=3)
```

## Applications

- Portfolio diversification
- Pairs trading identification
- Risk management
- Hedge effectiveness
- Market regime detection

---

## Running it

```bash
cd "Finance - Correlation Analysis"
python "correlation_analysis.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
