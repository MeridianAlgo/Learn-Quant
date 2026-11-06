# Merton Credit Risk Model

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Credit%20Risk">source</a></p>


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

## Running it

```bash
cd "Finance - Credit Risk"
python "merton_model.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
