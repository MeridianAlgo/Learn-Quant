"""
Cross Validation
----------------
A model scored on the data it was trained on is a student grading their own
exam. Cross validation is the discipline of always holding some data back,
fitting on one part and judging on another, so the score reflects how the
model will behave on data it has never seen.

This lesson builds the machinery from scratch on NumPy. A simple train and
test split, k fold indices that rotate the held out block around the data set,
a scorer that averages the folds, and finally walk forward splits for time
series, where shuffling would let the model peek into its own future. That
last point is the one that bites in quant work, ordinary k fold quietly leaks
tomorrow's prices into today's training set.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and targets into a training part and a testing part.

    With shuffle on, rows are drawn at random, which is right for data where
    order carries no meaning. For time series pass shuffle False so the test
    set is the most recent block, the only data the model truly has not seen.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    n = len(x)
    n_test = max(1, int(round(n * test_fraction)))
    order = np.arange(n)
    if shuffle:
        order = np.random.default_rng(seed).permutation(n)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def kfold_indices(
    n: int, k: int = 5, shuffle: bool = True, seed: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k pairs of (train indices, test indices) covering the data once.

    The data is cut into k nearly equal folds and each fold takes one turn as
    the test set while the rest train. Every row is tested exactly once, so
    the average score uses all the data without ever scoring a row the model
    saw in training.
    """
    if k < 2:
        raise ValueError("need at least two folds")
    if k > n:
        raise ValueError("cannot have more folds than rows")
    order = np.arange(n)
    if shuffle:
        order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, k)
    pairs = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        pairs.append((train_idx, test_idx))
    return pairs


def cross_val_score(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    seed: int | None = None,
) -> list[float]:
    """Score a model across k folds and return the per fold mean squared errors.

    The model is passed in as a single function that takes training features,
    training targets, and test features, and returns predictions. Keeping the
    model behind one function means any estimator fits this harness, from a
    mean predictor to a neural network.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scores = []
    for train_idx, test_idx in kfold_indices(len(x), k, seed=seed):
        predictions = fit_predict(x[train_idx], y[train_idx], x[test_idx])
        errors = np.asarray(predictions, dtype=float) - y[test_idx]
        scores.append(float(np.mean(errors**2)))
    return scores


def walk_forward_splits(n: int, n_splits: int = 5, min_train: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding window splits where training always precedes testing.

    The first split trains on the earliest block and tests on the next one,
    then the training window grows and the test block slides forward. No test
    row ever comes before a training row, which is the property that makes
    the score honest on time series.
    """
    if n_splits < 1:
        raise ValueError("need at least one split")
    boundaries = np.linspace(min_train, n, n_splits + 1, dtype=int)
    if boundaries[0] < min_train or len(set(boundaries)) != len(boundaries):
        raise ValueError("not enough rows for that many splits")
    pairs = []
    for i in range(n_splits):
        train_idx = np.arange(boundaries[i])
        test_idx = np.arange(boundaries[i], boundaries[i + 1])
        pairs.append((train_idx, test_idx))
    return pairs


if __name__ == "__main__":
    print("Cross Validation")
    print("=" * 40)

    # A noisy linear relationship to learn.
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 10, size=(200, 1))
    y = 3.0 * x[:, 0] + 2.0 + rng.normal(0, 1.0, size=200)

    def linear_fit_predict(x_train, y_train, x_test):
        a = np.column_stack([x_train, np.ones(len(x_train))])
        coef, *_ = np.linalg.lstsq(a, y_train, rcond=None)
        return np.column_stack([x_test, np.ones(len(x_test))]) @ coef

    def mean_fit_predict(x_train, y_train, x_test):
        return np.full(len(x_test), y_train.mean())

    linear_scores = cross_val_score(linear_fit_predict, x, y, k=5, seed=0)
    mean_scores = cross_val_score(mean_fit_predict, x, y, k=5, seed=0)
    print(f"\n5 fold mean squared error, linear model  {np.mean(linear_scores):.3f}")
    print(f"5 fold mean squared error, mean baseline {np.mean(mean_scores):.3f}")

    print("\nWalk forward splits on 100 time ordered rows")
    for train_idx, test_idx in walk_forward_splits(100, n_splits=4, min_train=20):
        print(f"  train rows 0..{train_idx[-1]:3d}   test rows {test_idx[0]:3d}..{test_idx[-1]:3d}")
