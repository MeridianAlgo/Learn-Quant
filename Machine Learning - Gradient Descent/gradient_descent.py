"""
Gradient Descent
----------------
Gradient descent is the workhorse behind almost every model that learns. The
idea is small. You have a loss that measures how wrong your model is, and the
gradient of that loss points in the direction the error grows fastest. So you
take a step the other way, downhill, and repeat. Do that enough times with a
sensible step size and the model settles into the parameters that fit the data.

This lesson fits a plain linear model, predict y from X, by minimising the mean
squared error. It shows full batch descent, which uses every row to compute each
step, and stochastic descent, which uses one row at a time and is what scales to
large data. Everything runs on NumPy so you can see the maths without a learning
framework hiding it.
"""

from __future__ import annotations

import numpy as np


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Return the model prediction X w plus b for every row of X."""
    return X @ weights + bias


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared error, the average of the squared residuals.

    Squaring punishes large misses far more than small ones and keeps the loss
    smooth, which is exactly what gradient descent needs to follow.
    """
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(resid**2))


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rescale each column to zero mean and unit spread, return the scaled data.

    Descent struggles when features live on wildly different scales, because one
    large feature dominates the gradient. Standardising puts them on equal
    footing so a single learning rate works for all of them. Also returns the
    mean and standard deviation so you can apply the same transform to new data.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # a constant column would divide by zero
    return (X - mean) / std, mean, std


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 200,
) -> dict:
    """Fit a linear model by full batch gradient descent.

    Each epoch computes the gradient of the mean squared error over the whole
    dataset and nudges the weights and bias downhill by lr times that gradient.
    A learning rate that is too large overshoots and the loss blows up, one too
    small crawls. Returns the fitted weights, the bias, and the loss per epoch so
    you can plot the descent and confirm it is going down.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    weights = np.zeros(d)
    bias = 0.0
    history: list[float] = []

    for _ in range(epochs):
        y_pred = predict(X, weights, bias)
        error = y_pred - y
        # Gradients of the mean squared error with respect to weights and bias.
        grad_w = (2 / n) * (X.T @ error)
        grad_b = (2 / n) * error.sum()
        weights -= lr * grad_w
        bias -= lr * grad_b
        history.append(mse(y, predict(X, weights, bias)))

    return {"weights": weights, "bias": bias, "history": history}


def sgd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 50,
    seed: int = 0,
) -> dict:
    """Fit a linear model by stochastic gradient descent, one row per step.

    Instead of averaging the gradient over every row, stochastic descent updates
    from a single shuffled row at a time. Each step is noisier but far cheaper,
    and over a full pass the noise averages out. This is the version that scales
    to data too large to hold a full gradient over. Returns the same fields as
    the batch version, with one loss recorded per epoch.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    weights = np.zeros(d)
    bias = 0.0
    history: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(n)
        for i in order:
            xi = X[i]
            error = float(xi @ weights + bias - y[i])
            weights -= lr * 2 * error * xi
            bias -= lr * 2 * error
        history.append(mse(y, predict(X, weights, bias)))

    return {"weights": weights, "bias": bias, "history": history}


if __name__ == "__main__":
    print("Gradient Descent")
    print("=" * 40)

    # Build a linear truth y = 3 x1 minus 2 x2 plus 5, then add a little noise,
    # and check that descent recovers those coefficients.
    rng = np.random.default_rng(42)
    n = 400
    X_raw = rng.normal(size=(n, 2)) * np.array([10.0, 2.0])  # different scales on purpose
    true_w = np.array([3.0, -2.0])
    true_b = 5.0
    y = X_raw @ true_w + true_b + rng.normal(scale=0.5, size=n)

    Xs, mean, std = standardize(X_raw)

    batch = gradient_descent(Xs, y, lr=0.1, epochs=300)
    print("\nBatch gradient descent on standardised features")
    print(f"  first epoch loss  {batch['history'][0]:.4f}")
    print(f"  final epoch loss  {batch['history'][-1]:.6f}")
    # Map the learned weights back to the original unscaled feature space.
    recovered = batch["weights"] / std
    print(f"  recovered slopes  {recovered.round(3)}   true  {true_w}")

    stoch = sgd(Xs, y, lr=0.02, epochs=40)
    print("\nStochastic gradient descent")
    print(f"  final epoch loss  {stoch['history'][-1]:.6f}")

    print("\nLearning rate matters")
    for lr in [0.01, 0.1, 0.5]:
        run = gradient_descent(Xs, y, lr=lr, epochs=100)
        print(f"  lr {lr:<5} final loss {run['history'][-1]:.6f}")
