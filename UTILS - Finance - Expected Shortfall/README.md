# Expected Shortfall (CVaR)

Expected Shortfall (ES), also called Conditional Value at Risk (CVaR), measures the **expected loss given that losses exceed the VaR threshold**. It is a coherent risk measure — unlike VaR, it captures tail severity, not just frequency.

## Functions

| Function | Description |
|---|---|
| `historical_es(returns, confidence_level)` | Non-parametric ES from actual distribution |
| `parametric_es(returns, confidence_level)` | Normal-assumption ES |
| `cornish_fisher_es(returns, confidence_level)` | Skewness/kurtosis-adjusted ES |
| `es_summary(returns, confidence_level)` | All three estimates in one dict |

## Key Concepts

- **VaR vs ES**: VaR says "you won't lose more than X with 95% probability." ES says "given you exceed VaR, your average loss is Y."
- **Coherence**: ES satisfies subadditivity — diversification always reduces risk. VaR does not.
- **Cornish-Fisher**: Adjusts the normal quantile using higher moments. Better for fat-tailed (leptokurtic) returns.

## Example

```python
from expected_shortfall import es_summary
import numpy as np

returns = np.random.normal(0.001, 0.02, 252)
summary = es_summary(returns, confidence_level=0.95)
print(summary)
# {'historical_es': 0.0412, 'parametric_es': 0.0398, 'cornish_fisher_es': 0.0405, ...}
```

## When to Use

- Portfolio risk reporting (ES is required under Basel III / FRTB)
- Comparing risk across strategies with different tail behaviors
- Stress testing alongside VaR
