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
