# CAPM — Capital Asset Pricing Model

CAPM is the idea that won a Nobel Prize and still anchors how the industry
thinks about return and risk. Its claim is bold and simple: **the only risk the
market pays you to bear is systematic risk** — how much an asset moves with the
market (its *beta*). Diversifiable, asset-specific risk earns you nothing,
because you could have diversified it away for free.

```
Expected return = R_f + beta * (R_m - R_f)
```

`R_f` is the risk-free rate and `(R_m - R_f)` is the market risk premium — the
extra return you earn for each unit of market risk (beta) you take on.

## Functions

| Function | Description |
|---|---|
| `capm_expected_return(risk_free_rate, beta, market_return)` | CAPM fair return (scalar or array) |
| `jensens_alpha(actual_return, risk_free_rate, beta, market_return)` | Realised return minus CAPM-required return |
| `estimate_beta(asset_returns, market_returns)` | Beta = Cov(asset, market) / Var(market) |
| `security_market_line(betas, risk_free_rate, market_return)` | Fair return across a range of betas |

## The pieces

- **Risk-free rate (`R_f`)** — what you earn with no risk, e.g. a short-dated
  government bill.
- **Beta (`b`)** — sensitivity to the market. `b = 1` moves with the market,
  `b > 1` amplifies it, `b < 1` dampens it, `b < 0` hedges it.
- **Market risk premium (`R_m - R_f`)** — the extra return investors demand for
  holding the risky market over the risk-free asset.

## Example

```python
from capm_calculator import capm_expected_return, jensens_alpha, estimate_beta

rf, rm = 0.03, 0.09
print(capm_expected_return(rf, beta=1.2, market_return=rm))   # fair return

# Did a 14% return with beta 1.2 represent skill?
print(jensens_alpha(0.14, rf, 1.2, rm))                       # Jensen's alpha

# Estimate beta from return histories
print(estimate_beta(asset_returns, market_returns))
```

## Jensen's alpha — skill or just risk?

A 20% return sounds great until you learn it came from a beta-2 bet in a bull
market — CAPM would have *expected* that. **Jensen's alpha** strips out the
return you were owed for the risk taken and leaves only the unexplained excess:

```
alpha = actual_return - [ R_f + beta * (R_m - R_f) ]
```

Persistent positive alpha is the holy grail of active management — and, CAPM
purists argue, usually a sign of a missing risk factor rather than magic.

## The security market line (SML)

Plot expected return against beta and CAPM is a straight line from `R_f` (at
`beta = 0`) through the market portfolio (at `beta = 1`). Assets **above** the
line are under-priced (more return than their risk warrants); assets **below**
are over-priced.

## Practical notes

- Beta is **estimated**, not given — it is the regression slope of the asset on
  the market and drifts over time. See `Finance - Beta Calculator` for rolling
  estimates, and `Quantitative Methods - Regression Analysis` for the mechanics.
- CAPM is a *single-factor* model. Real returns need more factors (size, value,
  momentum) — continue to `Quantitative Methods - Factor Models`.
- Use a consistent period: annual `R_f` and `R_m` with annual beta, or scale
  everything to the same frequency.
- CAPM underpins the discount rate in `Discounted Cash Flow (DCF)` and the
  appraisal ratio in `Finance - Information Ratio`.
