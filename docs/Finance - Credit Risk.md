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
