"""
K Nearest Neighbors
-------------------
Most models compress a training set into parameters and then throw the data
away. K nearest neighbors refuses to do that. It keeps every training point
and, when asked about a new one, simply looks up the k closest examples it
has already seen and lets them vote. There is nothing to fit, which is why it
is called a lazy learner, and all of the work happens at prediction time.

The only real decisions are how many neighbors to consult, how to measure
distance, and whether closer neighbors should count for more. Small k follows
the data closely and picks up noise, large k smooths everything toward the
overall average. Because distance is measured in raw feature units, a column
measured in thousands will drown out a column measured in fractions, so
standardizing the features first is not optional in practice.
"""

from __future__ import annotations

import numpy as np


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the features rescaled to zero mean and unit spread.

    Also returns the mean and the scale so the same transform can be applied
    to test data later. Columns with no variation get a scale of one so the
    division stays safe.
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return (x - mean) / scale, mean, scale


class KNearestNeighbors:
    """A k nearest neighbors predictor for classification or regression.

    Set `task` to "classification" for a majority vote over neighbor labels
    or "regression" for an average of neighbor values. With
    `weights="distance"` each neighbor counts in inverse proportion to how
    far away it is, which softens the jump that happens when one neighbor
    swaps for another.
    """

    def __init__(
        self,
        k: int = 5,
        task: str = "classification",
        weights: str = "uniform",
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least one")
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.task = task
        self.weights = weights
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: list[list[float]], y: list[float]) -> KNearestNeighbors:
        """Store the training data, which is the whole of the training step."""
        x_array = np.atleast_2d(np.asarray(x, dtype=float))
        y_array = np.asarray(y)
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if x_array.shape[0] < self.k:
            raise ValueError("need at least k training points")
        self.x_train = x_array
        self.y_train = y_array
        self.classes_ = np.unique(y_array) if self.task == "classification" else None
        return self

    def _neighbors(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the labels and distances of the k closest training points."""
        distances = np.linalg.norm(self.x_train - point, axis=1)
        # argpartition finds the k smallest without sorting the whole array.
        idx = np.argpartition(distances, self.k - 1)[: self.k]
        return self.y_train[idx], distances[idx]

    def _vote_weights(self, distances: np.ndarray) -> np.ndarray:
        """Return the weight each neighbor carries in the vote."""
        if self.weights == "uniform":
            return np.ones_like(distances)
        # An exact hit would divide by zero, so let it own the whole vote.
        if np.any(distances == 0):
            return (distances == 0).astype(float)
        return 1.0 / distances

    def predict(self, x: list[list[float]]) -> np.ndarray:
        """Return a prediction for every row of `x`."""
        if self.x_train is None:
            raise ValueError("call fit before predict")
        points = np.atleast_2d(np.asarray(x, dtype=float))
        if points.shape[1] != self.x_train.shape[1]:
            raise ValueError("x has a different number of features than the training data")

        predictions = []
        for point in points:
            labels, distances = self._neighbors(point)
            weight = self._vote_weights(distances)
            if self.task == "regression":
                predictions.append(float(np.sum(weight * labels) / np.sum(weight)))
            else:
                totals = [weight[labels == c].sum() for c in self.classes_]
                predictions.append(self.classes_[int(np.argmax(totals))])
        return np.array(predictions)

    def predict_proba(self, x: list[list[float]]) -> np.ndarray:
        """Return the share of the neighbor vote each class received.

        This is a crude probability. With five neighbors it can only ever
        take five values, so treat it as a ranking signal rather than a
        calibrated number.
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only defined for classification")
        if self.x_train is None:
            raise ValueError("call fit before predict_proba")
        points = np.atleast_2d(np.asarray(x, dtype=float))

        rows = []
        for point in points:
            labels, distances = self._neighbors(point)
            weight = self._vote_weights(distances)
            totals = np.array([weight[labels == c].sum() for c in self.classes_])
            rows.append(totals / totals.sum())
        return np.vstack(rows)


def accuracy(y_true: list[float], y_pred: list[float]) -> float:
    """Return the fraction of predictions that match the truth."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


if __name__ == "__main__":
    print("K Nearest Neighbors")
    print("=" * 40)

    rng = np.random.default_rng(3)

    # Two overlapping clouds standing in for up days and down days.
    up = rng.normal([1.0, 1.0], 1.0, size=(200, 2))
    down = rng.normal([-1.0, -1.0], 1.0, size=(200, 2))
    x = np.vstack([up, down])
    y = np.array([1] * 200 + [0] * 200)

    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    x_train, y_train = x[:300], y[:300]
    x_test, y_test = x[300:], y[300:]

    x_train_scaled, mean, scale = standardize(x_train)
    x_test_scaled = (x_test - mean) / scale

    print("\nneighbors  accuracy")
    for k in (1, 3, 15, 101, 299):
        model = KNearestNeighbors(k=k).fit(x_train_scaled, y_train)
        score = accuracy(y_test, model.predict(x_test_scaled))
        print(f"  {k:3d}       {score:.3f}")

    print("\nk of one memorises the training noise. Once k approaches the size of")
    print("the training set every vote returns the same answer and the boundary")
    print("disappears entirely.")

    weighted = KNearestNeighbors(k=15, weights="distance").fit(x_train_scaled, y_train)
    print(f"\ndistance weighted k of 15  {accuracy(y_test, weighted.predict(x_test_scaled)):.3f}")
