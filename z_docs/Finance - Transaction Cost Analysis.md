<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Transaction Cost Analysis"
    python "tca_utils.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Transaction%20Cost%20Analysis)

---
# Transaction Cost Analysis (TCA)

Tools for measuring execution quality and estimating market impact. TCA is essential for evaluating whether a strategy's theoretical alpha survives real-world trading costs.

## Functions

| Function | Description |
|---|---|
| `vwap(prices, volumes)` | Volume Weighted Average Price |
| `twap(prices)` | Time Weighted Average Price |
| `vwap_slippage(exec_price, vwap, side)` | Slippage vs. VWAP in bps |
| `implementation_shortfall(decision_price, ...)` | IS components vs. arrival price |
| `almgren_chriss_impact(order_size, adv, sigma, T, ...)` | Linear impact model |
| `sqrt_market_impact(order_size, adv, sigma, alpha)` | Empirical square-root rule |

## Key Concepts

### VWAP Benchmark
The most common execution benchmark. Trading algorithms attempt to match VWAP over a period. Slippage = `(exec - VWAP) / VWAP * 10,000` bps.

### Implementation Shortfall
More rigorous than VWAP. Measures the cost of the **entire decision** from signal to completion:
- `IS = (avg_execution - decision_price) / decision_price`
- Also captures missed opportunity cost for partially filled orders.

### Market Impact
- **Temporary impact**: Immediate price pressure from order flow, reverting after trade
- **Permanent impact**: Lasting information-based price move
- **Square-root rule**: `Impact ∝ sigma × sqrt(participation_rate)` — empirically robust across markets

## Example

```python
from tca_utils import implementation_shortfall, almgren_chriss_impact

# IS calculation
is_result = implementation_shortfall(
    decision_price=100.00,
    execution_prices=[100.05, 100.10, 100.15],
    execution_quantities=[1000, 1000, 1000],
    final_price=100.25,
)

# Impact for 100k share order in 1M ADV stock over 5 days
impact = almgren_chriss_impact(100_000, 1_000_000, sigma=0.015, T=5)
```

## Practical Rule of Thumb

For liquid large-caps: 1% of ADV ≈ 5–15 bps of impact. 10% of ADV ≈ 30–60 bps.


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
