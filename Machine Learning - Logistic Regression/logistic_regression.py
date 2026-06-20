"""
Logistic Regression
--------------------
Linear regression predicts a number; **logistic regression** predicts a
*probability* — the chance an example belongs to the positive class. In quant
work that is exactly the question behind "will tomorrow be an up day?",
"will this loan default?", or "is this order likely to be toxic flow?".

It is the simplest useful classifier and the gateway to everything else: it fits
a linear score ``z = X·w + b`` and squashes it through the **sigmoid** into
``(0, 1)``. Training minimises the **log-loss** (cross-entropy) by gradient
descent — the loss is convex, so plain gradient descent finds the global optimum
without the local-minima worries of larger models.

This module implements the fit, the gradients, L2 regularisation and the usual
metrics from scratch with NumPy, so nothing is hidden behind a library call.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sigmoid(z) -> np.ndarray:
    """The logistic function ``1 / (1 + e^-z)``, mapping any real to ``(0, 1)``.

    Written piecewise to stay numerically stable for large-magnitude ``z``
    (a naive ``exp`` overflows for very negative inputs).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(y, p, eps: float = 1e-15) -> float:
    """Mean binary cross-entropy between labels *y* and predicted probabilities *p*.

    This is the quantity training minimises: it punishes confident wrong
    predictions harshly (a predicted 0.99 on a true 0 costs a lot).
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit(
    X,
    y,
    lr: float = 0.1,
    epochs: int = 1000,
    l2: float = 0.0,
    random_state: int = 0,
) -> Tuple[np.ndarray, float]:
    """Fit logistic-regression weights by batch gradient descent.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Binary labels (0/1) of length ``n_samples``.
        lr: Learning rate (step size).
        epochs: Number of full-batch gradient steps.
        l2: L2 (ridge) penalty strength; 0 disables it. The bias is not penalised.
        random_state: Seed for the tiny weight initialisation.

    Returns:
        ``(weights, bias)`` — a length-``n_features`` vector and a scalar.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    w = rng.normal(0.0, 0.01, size=d)
    b = 0.0

    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        error = p - y  # gradient of log-loss wrt the linear score
        grad_w = X.T @ error / n + l2 * w
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, float(b)


def predict_proba(X, w, b) -> np.ndarray:
    """Predicted positive-class probabilities for each row of *X*."""
    return sigmoid(np.asarray(X, dtype=float) @ w + b)


def predict(X, w, b, threshold: float = 0.5) -> np.ndarray:
    """Hard 0/1 predictions by thresholding the probabilities (default 0.5)."""
    return (predict_proba(X, w, b) >= threshold).astype(int)


def accuracy(y_true, y_pred) -> float:
    """Fraction of labels predicted correctly."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def standardize(X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column; returns ``(X_scaled, mean, std)`` for reuse on test data.

    Gradient descent converges far faster on standardised features, and L2
    regularisation only makes sense when features share a scale.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


if __name__ == "__main__":
    print("Logistic Regression")
    print("=" * 40)

    # Two Gaussian blobs: one centred low, one high -> linearly separable-ish.
    rng = np.random.default_rng(0)
    n = 200
    X0 = rng.normal([-1.5, -1.5], 1.0, size=(n, 2))
    X1 = rng.normal([1.5, 1.5], 1.0, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    Xs, mu, sd = standardize(X)
    w, b = fit(Xs, y, lr=0.5, epochs=2000, l2=0.01)

    p = predict_proba(Xs, w, b)
    preds = predict(Xs, w, b)
    print(f"Trained on {len(X)} points, 2 features")
    print(f"Weights: {np.round(w, 3)}  bias: {b:.3f}")
    print(f"Log-loss : {log_loss(y, p):.4f}")
    print(f"Accuracy : {accuracy(y, preds):.1%}")

    # Classify a fresh point (remember to scale with the training stats).
    new = (np.array([[2.0, 2.0]]) - mu) / sd
    print(f"\nP(up | point near the high blob) = {predict_proba(new, w, b)[0]:.3f}")
