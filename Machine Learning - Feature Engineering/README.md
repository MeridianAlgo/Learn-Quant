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
