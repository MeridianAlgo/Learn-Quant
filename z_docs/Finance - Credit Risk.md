<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Credit Risk"
    python "merton_model.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Credit%20Risk)

---
# Merton Credit Risk Model

The Merton (1974) structural credit model treats a firm's **equity as a call option on its assets**. Default occurs when asset value falls below debt face value at maturity.

## Functions

| Function | Description |
|---|---|
| `merton_equity(V, F, r, sigma_V, T)` | Equity value via Black-Scholes formula |
| `merton_model(V, F, r, sigma_V, T)` | Full analytics: DD, PD, credit spread |
| `implied_asset_value(E, F, r, sigma_E, T)` | Back out asset value from observable equity |

## Key Outputs

- **Distance to Default (DD)**: How many standard deviations the firm is from the default threshold. DD > 3 is considered safe; DD < 1 is distressed.
- **Probability of Default (PD)**: `PD = N(-DD)`. Risk-neutral default probability.
- **Credit Spread**: `yield_on_debt - risk_free_rate`. Excess yield investors demand for bearing default risk.

## The Intuition

```
Equity  = Call(Assets, Strike=Debt, T=Maturity)
Debt    = Assets - Equity  (bondholders own residual if equity worthless)
Default = Assets < Debt at maturity
```

## Example

```python
from merton_model import merton_model

result = merton_model(V=100e6, F=80e6, r=0.05, sigma_V=0.20, T=1.0)
print(f"PD: {result['probability_of_default']:.2%}")
print(f"Credit Spread: {result['credit_spread_bps']:.1f} bps")
```

## Limitations

- Assumes simple capital structure (one class of debt with fixed maturity)
- Asset value and volatility are unobservable — must be inferred from equity
- Better suited for investment-grade firms; CDS-based models preferred for distressed credits


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
