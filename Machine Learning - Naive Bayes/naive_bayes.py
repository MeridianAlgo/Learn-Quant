"""
Naive Bayes
-----------
Bayes' theorem tells you how to update a belief about a class label after
seeing evidence. The naive part is a bold simplification, assume every
feature is independent of every other feature once you know the class. That
assumption is almost always false, yet the resulting classifier is fast,
needs little data, and is surprisingly hard to beat as a baseline.

The Gaussian flavour implemented here models each feature within each class
as a normal distribution. Training just means computing a mean and a variance
per feature per class plus the class frequencies. Prediction multiplies the
class prior by each feature's likelihood and picks the biggest product, done
in log space so hundreds of tiny probabilities do not underflow to zero.
"""

from __future__ import annotations

import numpy as np


def fit_gaussian_nb(features: np.ndarray, labels: np.ndarray) -> dict:
    """Learn per class priors, means, and variances from labelled data.

    Returns a model dictionary holding the sorted class labels, the log prior
    of each class, and a mean and variance per feature per class. A small
    floor is added to every variance so a constant feature cannot produce a
    division by zero at prediction time.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError("features must be a 2d array of shape (samples, features)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same number of rows")

    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("need at least two classes")

    means, variances, log_priors = [], [], []
    var_floor = 1e-9 * float(np.var(X)) + 1e-12
    for c in classes:
        rows = X[y == c]
        means.append(rows.mean(axis=0))
        variances.append(rows.var(axis=0) + var_floor)
        log_priors.append(np.log(rows.shape[0] / X.shape[0]))
    return {
        "classes": classes,
        "log_priors": np.asarray(log_priors),
        "means": np.asarray(means),
        "variances": np.asarray(variances),
    }


def log_posteriors(model: dict, features: np.ndarray) -> np.ndarray:
    """Return the unnormalized log posterior of each class for each sample.

    For every class this adds the log prior to the sum of Gaussian log
    likelihoods across features. Working in logs turns the product of many
    small numbers into a stable sum.
    """
    X = np.atleast_2d(np.asarray(features, dtype=float))
    out = np.empty((X.shape[0], model["classes"].size))
    for k in range(model["classes"].size):
        mean = model["means"][k]
        var = model["variances"][k]
        log_lik = -0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var)
        out[:, k] = model["log_priors"][k] + log_lik.sum(axis=1)
    return out


def predict(model: dict, features: np.ndarray) -> np.ndarray:
    """Return the most probable class label for each sample."""
    scores = log_posteriors(model, features)
    return model["classes"][np.argmax(scores, axis=1)]


def predict_proba(model: dict, features: np.ndarray) -> np.ndarray:
    """Return normalized class probabilities for each sample.

    The log posteriors are shifted by their row maximum before
    exponentiating, the standard trick that keeps the softmax stable.
    """
    scores = log_posteriors(model, features)
    shifted = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    return p / p.sum(axis=1, keepdims=True)


def accuracy(model: dict, features: np.ndarray, labels: np.ndarray) -> float:
    """Return the fraction of samples the model labels correctly."""
    return float(np.mean(predict(model, features) == np.asarray(labels)))


if __name__ == "__main__":
    print("Gaussian Naive Bayes")
    print("=" * 40)

    rng = np.random.default_rng(3)

    # Two market regimes, calm days and stressed days, described by two
    # features, absolute return and volume ratio.
    calm = np.column_stack([rng.normal(0.3, 0.15, 300), rng.normal(1.0, 0.2, 300)])
    stress = np.column_stack([rng.normal(1.2, 0.5, 300), rng.normal(1.8, 0.4, 300)])
    X = np.vstack([calm, stress])
    y = np.array(["calm"] * 300 + ["stress"] * 300)

    model = fit_gaussian_nb(X, y)
    print(f"\ntraining accuracy {accuracy(model, X, y):.1%}")

    day = np.array([[0.9, 1.5]])
    probs = predict_proba(model, day)[0]
    print("\nA day with a 0.9 percent move and 1.5x volume")
    for c, p in zip(model["classes"], probs):
        print(f"  {c:8s} {p:.1%}")
    print(f"  call {predict(model, day)[0]}")
