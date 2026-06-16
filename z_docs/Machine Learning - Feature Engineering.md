<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">AI & Machine Learning</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Machine Learning - Feature Engineering"
    python "feature_engineering.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Machine%20Learning%20-%20Feature%20Engineering)

---
# Feature Engineering for Financial ML

The dirty secret of quant machine learning: the model is rarely the bottleneck.
A random forest and a neural net will perform almost identically on *good*
features and *honest* labels — and both will fail on raw prices. This module is
about the part that actually moves the needle: turning a price series into a
clean, stationary feature matrix and well-posed labels, without leaking the
future into the past.

## Functions

| Function | Description |
|---|---|
| `make_features(prices, windows)` | Build returns, momentum, volatility, z-score, RSI and SMA-distance features |
| `forward_return_label(prices, horizon)` | Simple forward-return and direction (+1/-1) labels |
| `triple_barrier_labels(prices, pt, sl, max_holding)` | Profit-take / stop-loss / time-limit labelling |
| `purged_train_test_split(features, labels, test_size, embargo)` | Leak-free chronological split with an embargo gap |

## Why not just feed prices to a model?

Prices are **non-stationary**: their mean and variance drift, so a level a model
learned in 2015 is meaningless in 2024. Engineered features fix this by being
*relative* and *bounded*:

- **Returns** (`ret_1`) — stationary by construction.
- **Momentum** (`mom_w`) — cumulative return over a window; trend signal.
- **Volatility** (`vol_w`) — rolling standard deviation of returns.
- **Z-score** (`zscore_w`) — how stretched price is vs. its own recent mean;
  mean-reversion signal.
- **RSI** (`rsi_14`) — Wilder's bounded oscillator.
- **SMA distance** (`dist_sma_w`) — price relative to a moving average.

## Labelling: the triple-barrier method

A naive "did price go up in N days?" label ignores *how* it got there. The
**triple-barrier method** (López de Prado) labels a hypothetical trade by which
barrier it touches first:

```
        +pt  ── profit-take  → label +1
entry ──┤
        -sl  ── stop-loss    → label -1
         |
     max_holding bars later  → label 0 (time barrier)
```

This produces labels that match how a position is actually managed — with a
target and a stop — and gives a far more realistic learning target than a fixed
horizon return.

## Example

```python
import pandas as pd
from feature_engineering import (
    make_features, triple_barrier_labels, purged_train_test_split,
)

prices = pd.Series(...)                 # your price series
X = make_features(prices)
y = triple_barrier_labels(prices, pt=0.02, sl=0.02, max_holding=10)["label"]

X_tr, y_tr, X_te, y_te = purged_train_test_split(X, y, test_size=0.3, embargo=5)
# ... fit your model on (X_tr, y_tr), evaluate on (X_te, y_te)
```

## Avoiding the cardinal sin: leakage

The fastest way to a backtest that prints money and then loses it live is
**look-ahead leakage** — letting test-time data influence training. Two guards
here:

1. Every feature is **backward-looking** (rolling windows, lags) — no value uses
   information from after the bar it is attached to.
2. `purged_train_test_split` splits **chronologically** and inserts an
   **embargo** gap, so a test row's look-back window cannot overlap the training
   set.

## Practical notes

- Feed the output straight into `Machine Learning - Random Forest` or
  `Machine Learning Time Series`.
- Triple-barrier labels are usually imbalanced — check the class balance (the
  demo prints it) and weight your model or resample accordingly.
- More features is not better. Many of these are correlated (momentum vs.
  SMA-distance); prune with feature importance and watch for multicollinearity.
- Standardise features (e.g. with the training-set mean/std only) before feeding
  distance-based models.


---

## Continue in AI & Machine Learning

<div class="grid cards" markdown>

-   :material-robot-outline: __[AI Development](AI Development.md)__

    Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js. This module demonstrates how to integrate a hosted large language model into a simple interactive application.

-   :material-robot-outline: __[Learning Platform](Learning Platform.md)__

    An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

-   :material-robot-outline: __[Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md)__

    Given a few hundred stocks and their return characteristics, which ones behave

-   :material-robot-outline: __[Machine Learning - Random Forest](Machine Learning - Random Forest.md)__

    This module provides a basic implementation of a Random Forest Predictor for quantitative finance. It uses scikit-learn's `RandomForestRegressor` to predict time series data or returns based on a set of features.

-   :material-robot-outline: __[Machine Learning Time Series](Machine Learning Time Series.md)__

    Applying incredibly sophisticated statistical and advanced computational matrix calculating algorithms to historical sequential asset prices explicitly enables quantitative researchers to discover heavily latent non linear correlation patterns. Standard basic linear techniques lack the internal theoretical mapping memory required to fully process continuous progression data natively. Therefore, explicit sequential data pattern prediction necessitates deeply specialized memory architectures uniquely capable of successfully retaining vast contextual numerical memory safely across thousands of chronologically independent market observations simultaneously.

-   :material-robot-outline: __[Reinforcement Learning Q Learning](Reinforcement Learning Q Learning.md)__

    This module extensively covers the core mathematical algorithms necessary to construct entirely autonomous quantitative execution agents. Rather than relying on rigid statistical parameters or explicit condition based trading logic, reinforcement learning allows an agent to discover the most optimal sequences of action through continuous simulated trial and error. The intelligent agent dynamically interprets complex environmental states and receives explicit scalar rewards or punitive penalties based directly upon its transactional profitability and risk management threshold maintenance. Over thousands of episodes, the model organically maps the market mechanics to develop a mathematically optimal trading policy without human intervention.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
