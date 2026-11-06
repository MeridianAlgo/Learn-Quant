# Cointegration & Pairs Trading Foundations

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Cointegration">source</a></p>


Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

## Functions

| Function | Description |
|---|---|
| `ols_hedge_ratio(y, x)` | OLS regression for hedge ratio + residuals |
| `adf_test(series, lags)` | Augmented Dickey-Fuller unit-root test |
| `engle_granger(y, x, lags)` | Two-step cointegration test |
| `half_life(spread)` | Mean-reversion half-life via OU fit |
| `zscore_spread(spread, window)` | Rolling z-score for entry/exit signals |

## ADF Critical Values (one-sided)

| Significance | Critical t-stat |
|---|---|
| 1% | -3.43 |
| 5% | -2.86 |
| 10% | -2.57 |

t-stat **more negative** than the threshold → reject unit root → stationary.

## Example

```python
from cointegration import engle_granger, half_life, zscore_spread

eg = engle_granger(prices_a, prices_b, lags=1)
if eg['cointegrated_5pct']:
    spread = eg['spread']
    z = zscore_spread(spread, window=60)
    hl = half_life(spread)
    # Enter when |z| > 2, exit when |z| < 0.5
```

## Practical Notes

- Half-life > 30 days → spread too slow, transaction costs eat profits.
- Half-life < 1 day → likely noise, not true mean reversion.
- Cointegration relationships **break** — re-test rolling windows.
- For >2 assets use Johansen's test (not implemented here).

---

## Running it

```bash
cd "Quantitative Methods - Cointegration"
python "cointegration.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md): This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.
- [Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md): Factor models explain asset returns as a linear combination of systematic factors plus a stock-specific residual.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
