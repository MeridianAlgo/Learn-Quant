<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Regime Detection"
    python "regime_detection.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Regime%20Detection)

---
# Market Regime Detection

Identifies distinct market states (bull/bear, low/high volatility) using statistical methods. Regime-aware strategies adapt parameters to the current market environment.

## Functions

| Function | Description |
|---|---|
| `moving_average_regime(prices, short, long)` | MA crossover bull/bear detection |
| `volatility_regime(returns, window, n_regimes)` | Quantile-based volatility buckets |
| `gaussian_mixture_regime(returns, n_regimes)` | GMM-based unsupervised regime detection |
| `regime_stats(returns, labels)` | Per-regime return statistics |

## Methods

### Moving Average Crossover
Classic technical approach: Bull when 50-day MA > 200-day MA (golden cross), Bear otherwise. Simple, interpretable, but lagging.

### Volatility Regime
Rolling realized volatility classified into low/medium/high buckets using quantile thresholds. Useful for dynamic position sizing.

### Gaussian Mixture Model (GMM)
Unsupervised learning: fit a mixture of Gaussians to the return distribution. Regime 0 = lowest mean (bear), Regime 1 = highest mean (bull). Requires `scikit-learn`.

## Example

```python
from regime_detection import gaussian_mixture_regime, regime_stats
import numpy as np

returns = np.random.normal(0.001, 0.015, 500)
result = gaussian_mixture_regime(returns, n_regimes=2)
stats = regime_stats(returns, result["labels"])
```

## Applications

- **Strategy switching**: Use momentum in bull regimes, mean-reversion in bear
- **Risk scaling**: Reduce position sizes in high-volatility regimes
- **Macro overlay**: Override signals when macro regime shifts


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md)__

    A strategy wins 7 of its first 10 trades. Is its true win rate 70%? Almost

-   :material-function-variant: __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

-   :material-function-variant: __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   :material-function-variant: __[Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)__

    This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

-   :material-function-variant: __[Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)__

    Most risk models assume returns are normally distributed. They are not —

-   :material-function-variant: __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
