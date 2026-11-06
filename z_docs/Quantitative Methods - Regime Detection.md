# Market Regime Detection

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Regime%20Detection">source</a></p>


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

## Running it

```bash
cd "Quantitative Methods - Regime Detection"
python "regime_detection.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md): Cointegration: two non-stationary series whose linear combination is stationary.
- [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md): This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
