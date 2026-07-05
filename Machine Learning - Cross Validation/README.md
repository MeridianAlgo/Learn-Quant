# Machine Learning, Cross Validation

A model scored on the data it was trained on is a student grading their own
exam. It will look brilliant right up to the moment it meets the real world.
Cross validation is the discipline of always holding some data back, fitting
on one part and judging on another, so the score you report is an estimate of
how the model behaves on data it has never seen.

This lesson builds the machinery from scratch on NumPy, no scikit learn
required, so you can see that there is nothing mysterious inside. It ends with
the version that matters most in quant work, walk forward validation, because
ordinary shuffled folds quietly let a model peek into its own future when the
rows are ordered in time.

## The pieces

* `train_test_split(x, y, test_fraction, shuffle, seed)` carves off a test set.
  Shuffled for ordinary data, unshuffled for time series so the test block is
  the most recent data.
* `kfold_indices(n, k, shuffle, seed)` cuts the row indices into k nearly equal
  folds and rotates each fold through the role of test set. Every row is tested
  exactly once and trained on k minus one times.
* `cross_val_score(fit_predict, x, y, k, seed)` runs the folds and returns the
  mean squared error of each one. The model goes in as a single function that
  takes training features, training targets, and test features, so any
  estimator fits the harness.
* `walk_forward_splits(n, n_splits, min_train)` produces expanding window
  splits where the training rows always come before the test rows.

## Why one split is not enough

A single train and test split gives you one number, and that number depends on
which rows happened to land in the test set. Run it again with a different
seed and the score moves. K fold averages over k different test sets, which
both steadies the estimate and shows you its spread. If the fold scores vary
wildly, that spread is itself a finding, the model is sensitive to which data
it sees.

## The time series trap

Shuffled k fold assumes the rows are exchangeable, that no row tells you about
its neighbours. Prices violate this on purpose, today's return is correlated
with yesterday's volatility and tomorrow's is correlated with today's. Shuffle
a return series into folds and the model trains on Wednesday while being
tested on Tuesday, a rehearsal of the answer sheet. Backtests validated this
way look wonderful and then fail in production. Walk forward splits fix it by
construction, training always ends before testing begins, exactly the
constraint live trading imposes.

## Example

```python
import numpy as np
from cross_validation import cross_val_score, walk_forward_splits

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, size=(100, 1))
y = 3 * x[:, 0] + rng.normal(0, 1, size=100)

def fit_predict(x_train, y_train, x_test):
    a = np.column_stack([x_train, np.ones(len(x_train))])
    coef, *_ = np.linalg.lstsq(a, y_train, rcond=None)
    return np.column_stack([x_test, np.ones(len(x_test))]) @ coef

scores = cross_val_score(fit_predict, x, y, k=5, seed=0)
print(np.mean(scores))

for train_idx, test_idx in walk_forward_splits(100, n_splits=4, min_train=20):
    print(len(train_idx), "training rows then", len(test_idx), "test rows")
```

## Where to go next

* For a model worth validating see
  [`Machine Learning - Logistic Regression`](../Machine%20Learning%20-%20Logistic%20Regression/)
  or [`Machine Learning - Random Forest`](../Machine%20Learning%20-%20Random%20Forest/).
* For the fitting engine underneath many models see
  [`Machine Learning - Gradient Descent`](../Machine%20Learning%20-%20Gradient%20Descent/).
* For the full backtesting discipline this generalises into see
  [`Strategies - Backtesting Engine`](../Strategies%20-%20Backtesting%20Engine/).
