<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Extreme Value Theory"
    python "extreme_value_theory.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Extreme%20Value%20Theory)

---
# Extreme Value Theory (EVT) for Tail Risk

Most risk models assume returns are normally distributed. They are not —
markets crash far more often than a bell curve allows. **Extreme Value Theory**
fixes this by modelling the *tail* of the loss distribution directly, instead of
forcing one distribution to fit both the calm middle and the violent edge.

This module implements the **Peaks-Over-Threshold (POT)** method and uses it to
compute Value-at-Risk and Expected Shortfall that respect fat tails.

## Functions

| Function | Description |
|---|---|
| `fit_gpd_mom(excesses)` | Fit a Generalised Pareto Distribution to threshold excesses (method of moments) |
| `pot_var_es(losses, confidence, threshold_quantile)` | EVT-based VaR and Expected Shortfall |
| `hill_estimator(data, k)` | Tail-index estimate from the top `k` order statistics |
| `mean_excess(losses, thresholds)` | Mean-excess function `e(u)` for threshold-selection diagnostics |

## The idea

The **Pickands–Balkema–de Haan theorem** says that for a wide class of
distributions, the excesses over a high threshold `u` converge to a
**Generalised Pareto Distribution (GPD)**:

```
G(y) = 1 - (1 + xi * y / beta)^(-1/xi),   y = loss - u > 0
```

- **xi** (shape / tail index): `xi > 0` ⇒ heavy tail (power-law). Equity losses
  typically sit around `xi ≈ 0.2–0.4`.
- **beta** (scale): sets the spread of the excesses.

Once the tail is fitted, VaR and ES follow in closed form
(McNeil, Frey & Embrechts):

```
VaR_q = u + (beta/xi) * [ (n/Nu * (1 - q))^(-xi) - 1 ]
ES_q  = VaR_q / (1 - xi) + (beta - xi*u) / (1 - xi)
```

where `n` is the sample size and `Nu` the number of excesses above `u`.

## Example

```python
import numpy as np
from extreme_value_theory import pot_var_es, hill_estimator

returns = ...                 # your return series
losses = -np.asarray(returns) # POT works on positive losses

evt = pot_var_es(losses, confidence=0.99, threshold_quantile=0.90)
print(evt["var"], evt["es"], evt["xi"])

# Heavy-tail diagnostic
print(hill_estimator(losses, k=200))
```

## Why it matters

At the 99.5% level a Gaussian model and an EVT model can disagree by a wide
margin — and the EVT number is almost always the larger, more realistic one.
The `__main__` demo fits both to fat-tailed Student-t returns so you can see the
gap the normal assumption hides.

## Practical notes

- **Threshold choice is the art.** Too low and you contaminate the tail with the
  body; too high and you have too few points to fit. The mean-excess plot
  (`mean_excess`) should be roughly linear above a good threshold; the `0.90`
  quantile default is a sensible starting point.
- **Expected Shortfall needs `xi < 1`.** If the fitted shape exceeds 1 the tail
  has no finite mean and ES is infinite — a sign the data (or threshold) needs
  another look.
- EVT complements the empirical and parametric methods in
  `Value at Risk (VaR)` and `Finance - Expected Shortfall`; use it specifically
  for the *deep* tail (99%+).
- The Hill estimator is simple but sensitive to `k`; plot `xi_hat` against `k`
  and look for a stable region.


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

-   :material-function-variant: __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   :material-function-variant: __[Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)__

    This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

-   :material-function-variant: __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

-   :material-function-variant: __[Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)__

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures **volatility clustering** — high-volatility days tend to follow high-volatility days. Used for risk forecasting, option pricing, and VaR.

-   :material-function-variant: __[Quantitative Methods - Interest Rate Models](Quantitative Methods - Interest Rate Models.md)__

    Continuous-time models for the evolution of the short (instantaneous) interest rate. Used for bond pricing, interest rate derivatives, and yield curve modeling.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
