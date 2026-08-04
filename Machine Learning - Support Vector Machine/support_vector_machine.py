"""
Support Vector Machine
----------------------
Most classifiers stop as soon as they find a line that separates the classes.
A support vector machine keeps going and looks for the line with the widest
empty corridor around it, on the argument that a boundary sitting far from
every training point is the one most likely to survive contact with data it
has never seen.

The points that end up touching the edges of that corridor are the support
vectors, and they are the only ones that matter. Move a point deep inside its
own class and the boundary does not budge. That is a very different failure
mode from a model where every observation tugs on the answer, and it is why a
support vector machine tends to be stubborn in the face of a few odd points
and yet sensitive to where exactly the classes meet.

Real data is rarely cleanly separable, so this implementation uses the soft
margin form. Points are allowed to sit inside the corridor or even on the
wrong side of it, and each one pays a penalty proportional to how far it
strayed. That penalty is the hinge loss, and training is nothing more than
sliding the boundary downhill until the total penalty stops falling.
"""

from __future__ import annotations

import numpy as np


def standardize(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Return standardized features along with the mean and deviation used.

    A support vector machine measures the width of its margin as a distance,
    so a feature carried in dollars will dominate one carried in percent for
    no reason other than the units. Standardizing first is not a nicety here,
    an unscaled fit is usually meaningless. The returned mean and deviation
    have to be reused on any future data, never recomputed from it.
    """
    data = np.asarray(X, dtype=float)
    if data.ndim != 2:
        raise ValueError("features must be a two dimensional array")
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # A constant column has no spread and dividing by its zero would explode.
    std = np.where(std == 0, 1.0, std)
    return ((data - mean) / std).tolist(), mean.tolist(), std.tolist()


def hinge_loss(scores: list[float], labels: list[int]) -> float:
    """Return the mean hinge loss for a set of raw decision scores.

    A point scoring above one on the correct side costs nothing at all, which
    is the whole point. Anything closer to the boundary than that, or on the
    wrong side entirely, costs the amount by which it fell short. The loss is
    zero only when every point clears the margin with room to spare.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if s.size != y.size:
        raise ValueError("scores and labels must be the same length")
    if s.size == 0:
        raise ValueError("need at least one observation")
    return float(np.mean(np.maximum(0.0, 1.0 - y * s)))


class LinearSVM:
    """A soft margin linear support vector machine trained by subgradient descent.

    The regularization strength `lam` sets the trade between a wide margin and
    a small penalty. Turn it up and the model prefers a broad corridor even if
    that means misclassifying a few points, turn it down and it will contort
    itself to get the training set right. It is the single knob worth tuning
    and the sensible way to tune it is cross validation.
    """

    def __init__(self, learning_rate: float = 0.01, lam: float = 0.01, epochs: int = 1000) -> None:
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if lam < 0:
            raise ValueError("regularization strength cannot be negative")
        if epochs < 1:
            raise ValueError("need at least one epoch")
        # The regularization step multiplies the weights by one minus this
        # product every epoch. Let it reach two and the weights flip sign and
        # grow without bound, and the fit quietly returns nonsense instead of
        # failing, so refuse the combination up front.
        if learning_rate * lam >= 2:
            raise ValueError("learning rate times regularization must stay below two or training diverges")
        self.learning_rate = learning_rate
        self.lam = lam
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]) -> LinearSVM:
        """Fit the boundary to standardized features and labels of plus or minus one.

        Labels given as zero and one are converted, since that encoding is
        common and the arithmetic of the hinge needs the symmetric version.
        The loss is recorded every epoch so you can see whether training
        actually settled or was still moving when it ran out of epochs.
        """
        data = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if data.ndim != 2:
            raise ValueError("features must be a two dimensional array")
        if data.shape[0] != labels.size:
            raise ValueError("need one label per row of features")
        unique = set(np.unique(labels).tolist())
        if unique <= {0, 1}:
            labels = np.where(labels == 1, 1.0, -1.0)
        elif unique <= {-1, 1}:
            labels = labels.astype(float)
        else:
            raise ValueError("labels must be either zero and one or minus one and one")
        if len(unique) < 2:
            raise ValueError("need both classes present to fit a boundary")

        n_samples, n_features = data.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            scores = data @ self.weights + self.bias
            # Only the points that fail to clear the margin push the boundary.
            violating = (labels * scores) < 1
            grad_w = self.lam * self.weights - (data[violating].T @ labels[violating]) / n_samples
            grad_b = -labels[violating].sum() / n_samples
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            penalty = 0.5 * self.lam * float(self.weights @ self.weights)
            self.loss_history.append(hinge_loss(scores.tolist(), labels.tolist()) + penalty)

        return self

    def decision_function(self, X: list[list[float]]) -> list[float]:
        """Return the signed distance of each row from the boundary, before scaling.

        The sign gives the predicted class and the magnitude gives confidence.
        A score beyond plus or minus one means the point sits outside the
        margin corridor entirely, which is where you want most of them.
        """
        if self.weights is None:
            raise ValueError("the model has not been fitted yet")
        data = np.asarray(X, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.weights.size:
            raise ValueError("features must have the same number of columns as the training data")
        return (data @ self.weights + self.bias).tolist()

    def predict(self, X: list[list[float]]) -> list[int]:
        """Return a class of plus or minus one for each row."""
        return np.where(np.asarray(self.decision_function(X)) >= 0, 1, -1).tolist()

    def support_vectors(self, X: list[list[float]], y: list[int], tolerance: float = 1e-6) -> list[int]:
        """Return the row indices of the points sitting on or inside the margin.

        These are the observations the boundary actually rests on. Everything
        else could be deleted from the training set and the fitted line would
        come out the same, which is worth checking once because it makes the
        idea concrete in a way the equations do not.
        """
        labels = np.asarray(y)
        if set(np.unique(labels).tolist()) <= {0, 1}:
            labels = np.where(labels == 1, 1.0, -1.0)
        scores = np.asarray(self.decision_function(X))
        return np.flatnonzero(labels * scores <= 1.0 + tolerance).tolist()

    def margin_width(self) -> float:
        """Return the width of the corridor between the two classes.

        The corridor runs from a score of minus one to a score of plus one, so
        in feature space its width is two divided by the length of the weight
        vector. Shrinking the weights is the same thing as widening the
        margin, which is exactly what the regularization term is doing.
        """
        if self.weights is None:
            raise ValueError("the model has not been fitted yet")
        norm = float(np.linalg.norm(self.weights))
        if norm == 0:
            return float("inf")
        return 2.0 / norm


def accuracy(predicted: list[int], actual: list[int]) -> float:
    """Return the fraction of labels predicted correctly, on either encoding."""
    p = np.asarray(predicted)
    a = np.asarray(actual)
    if p.size != a.size:
        raise ValueError("predicted and actual must be the same length")
    if a.size == 0:
        raise ValueError("need at least one observation")
    a = np.where(a == 1, 1, -1)
    return float(np.mean(p == a))


if __name__ == "__main__":
    print("Support Vector Machine")
    print("=" * 40)

    rng = np.random.default_rng(3)
    # Two groups of days described by momentum and volatility, one that went
    # on to rise and one that went on to fall.
    up = rng.normal([1.5, -0.8], 0.7, size=(120, 2))
    down = rng.normal([-1.2, 0.9], 0.7, size=(120, 2))
    features = np.vstack([up, down])
    targets = np.array([1] * 120 + [-1] * 120)

    scaled, mean, std = standardize(features.tolist())
    model = LinearSVM(learning_rate=0.05, lam=0.01, epochs=2000).fit(scaled, targets.tolist())

    print("\nFit on two hundred and forty labelled days")
    print(f"  training accuracy {accuracy(model.predict(scaled), targets.tolist()):.3f}")
    print(f"  margin width      {model.margin_width():.4f}")
    print(f"  support vectors   {len(model.support_vectors(scaled, targets.tolist()))} of {len(targets)}")

    print("\nLoss falling over training")
    for epoch in (0, 100, 500, 1999):
        print(f"  epoch {epoch:>4} {model.loss_history[epoch]:.5f}")

    print("\nWhat a wider margin costs")
    for lam in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = LinearSVM(learning_rate=0.05, lam=lam, epochs=2000).fit(scaled, targets.tolist())
        acc = accuracy(m.predict(scaled), targets.tolist())
        support = len(m.support_vectors(scaled, targets.tolist()))
        print(f"  lam {lam:<6} margin {m.margin_width():>8.4f}  support {support:>3}  accuracy {acc:.3f}")

    print("\nMore regularization widens the corridor until it swallows the whole")
    print("training set, at which point every point is a support vector and the")
    print("boundary is being set by the regularizer rather than by the data.")
