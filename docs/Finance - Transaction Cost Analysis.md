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
