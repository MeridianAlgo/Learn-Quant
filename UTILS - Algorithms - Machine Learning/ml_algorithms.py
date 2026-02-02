"""
Machine Learning Algorithms Implementation
A comprehensive collection of fundamental ML algorithms from scratch.
"""

from typing import Dict

import numpy as np


class LinearRegression:
    """
    Linear Regression using Gradient Descent.
    Finds best-fit line for linear relationship between features and target.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute loss (MSE)
            loss = np.mean((y_predicted - y) ** 2)
            self.loss_history.append(loss)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LogisticRegression:
    """
    Logistic Regression for binary classification.
    Uses sigmoid function for probability estimation.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) with values 0 or 1
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(
                y * np.log(y_predicted + 1e-15)
                + (1 - y) * np.log(1 - y_predicted + 1e-15)
            )
            self.loss_history.append(loss)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        y_predicted = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return (y_predicted > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates."""
        return self._sigmoid(np.dot(X, self.weights) + self.bias)


class KNN:
    """
    K-Nearest Neighbors classifier.
    Classifies based on majority vote of k nearest neighbors.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all samples."""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x: np.ndarray) -> int:
        """Predict for a single sample."""
        # Compute distances to all training points
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Majority vote
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


class KMeans:
    """
    K-Means clustering algorithm.
    Partitions data into k clusters by minimizing within-cluster variance.
    """

    def __init__(self, k: int = 3, max_iterations: int = 100, tolerance: float = 1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model to data.

        Args:
            X: Data points (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iterations):
            # Assign clusters
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array(
                [X[self.labels == j].mean(axis=0) for j in range(self.k)]
            )

            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign clusters to new data points."""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)


class DecisionTree:
    """
    Decision Tree for classification using ID3 algorithm.
    Uses information gain for splitting criteria.
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build the decision tree."""
        self.tree = self._build_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a set."""
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-15))

    def _information_gain(
        self, X: np.ndarray, y: np.ndarray, feature_idx: int
    ) -> float:
        """Calculate information gain for a feature."""
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Child entropies
        unique_values = np.unique(X[:, feature_idx])
        child_entropy = 0

        for value in unique_values:
            mask = X[:, feature_idx] == value
            child_y = y[mask]
            if len(child_y) > 0:
                child_entropy += (len(child_y) / len(y)) * self._entropy(child_y)

        return parent_entropy - child_entropy

    def _best_feature(self, X: np.ndarray, y: np.ndarray) -> int:
        """Find best feature to split on."""
        best_gain = -1
        best_feature = -1

        for feature_idx in range(X.shape[1]):
            gain = self._information_gain(X, y, feature_idx)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx

        return best_feature

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """Recursively build the decision tree."""
        # Stopping conditions
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(X) < self.min_samples_split
        ):
            return {"label": np.bincount(y).argmax()}

        # Find best feature to split
        best_feature = self._best_feature(X, y)

        if best_feature == -1:
            return {"label": np.bincount(y).argmax()}

        # Split data
        tree = {"feature": best_feature, "children": {}}
        unique_values = np.unique(X[:, best_feature])

        for value in unique_values:
            mask = X[:, best_feature] == value
            child_X = X[mask]
            child_y = y[mask]

            if len(child_y) > 0:
                tree["children"][value] = self._build_tree(child_X, child_y, depth + 1)

        return tree

    def _predict_single(self, x: np.ndarray, tree: Dict) -> int:
        """Predict for a single sample."""
        if "label" in tree:
            return tree["label"]

        feature_value = x[tree["feature"]]
        if feature_value in tree["children"]:
            return self._predict_single(x, tree["children"][feature_value])
        else:
            # Return most common label if feature value not seen during training
            return 0


class NaiveBayes:
    """
    Naive Bayes classifier.
    Uses Bayes' theorem with strong independence assumptions.
    """

    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.class_feature_means = None
        self.class_feature_vars = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # Initialize arrays
        self.class_priors = np.zeros(n_classes)
        self.class_feature_means = np.zeros((n_classes, n_features))
        self.class_feature_vars = np.zeros((n_classes, n_features))

        # Calculate statistics for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]

            # Prior probability
            self.class_priors[idx] = len(X_c) / len(X)

            # Mean and variance for each feature
            self.class_feature_means[idx] = np.mean(X_c, axis=0)
            self.class_feature_vars[idx] = (
                np.var(X_c, axis=0) + 1e-9
            )  # Avoid division by zero

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x: np.ndarray) -> int:
        """Predict for a single sample."""
        posteriors = []

        for i in range(len(self.classes)):
            # Log prior
            log_prior = np.log(self.class_priors[i])

            # Log likelihood (Gaussian)
            log_likelihood = np.sum(
                np.log(
                    self._gaussian_pdf(
                        x, self.class_feature_means[i], self.class_feature_vars[i]
                    )
                )
            )

            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _gaussian_pdf(
        self, x: np.ndarray, mean: np.ndarray, var: np.ndarray
    ) -> np.ndarray:
        """Gaussian probability density function."""
        return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def generate_sample_data():
    """Generate sample datasets for demonstration."""
    np.random.seed(42)

    # Linear regression data
    X_reg = np.random.randn(100, 1)
    y_reg = 2 * X_reg.ravel() + 1 + 0.1 * np.random.randn(100)

    # Classification data
    X_class = np.random.randn(200, 2)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)

    # Clustering data
    cluster1 = np.random.randn(50, 2) + np.array([2, 2])
    cluster2 = np.random.randn(50, 2) + np.array([-2, -2])
    cluster3 = np.random.randn(50, 2) + np.array([2, -2])
    X_cluster = np.vstack([cluster1, cluster2, cluster3])

    return X_reg, y_reg, X_class, y_class, X_cluster


def demonstrate_ml_algorithms():
    """Demonstrate all ML algorithms with sample data."""
    print("=== Machine Learning Algorithms Demonstration ===\n")

    # Generate sample data
    X_reg, y_reg, X_class, y_class, X_cluster = generate_sample_data()

    # Split data
    def train_test_split(X, y, test_size=0.2):
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    # Linear Regression
    print("--- Linear Regression ---")
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg)

    lr = LinearRegression(learning_rate=0.1, n_iterations=1000)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)

    print(f"R² Score: {score:.4f}")
    print(f"Final Loss: {lr.loss_history[-1]:.4f}")
    print()

    # Logistic Regression
    print("--- Logistic Regression ---")
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class)

    log_reg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Final Loss: {log_reg.loss_history[-1]:.4f}")
    print()

    # KNN
    print("--- K-Nearest Neighbors ---")
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class)

    knn = KNN(k=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Accuracy: {accuracy:.4f}")
    print()

    # K-Means
    print("--- K-Means Clustering ---")
    kmeans = KMeans(k=3)
    kmeans.fit(X_cluster)

    labels = kmeans.predict(X_cluster)
    print(f"Cluster assignments: {np.bincount(labels)}")
    print(f"Centroids shape: {kmeans.centroids.shape}")
    print()

    # Decision Tree
    print("--- Decision Tree ---")
    # Discretize features for decision tree
    X_discrete = np.digitize(X_class, bins=np.percentile(X_class, [33, 66]))
    X_train, X_test, y_train, y_test = train_test_split(X_discrete, y_class)

    dt = DecisionTree(max_depth=5, min_samples_split=5)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Accuracy: {accuracy:.4f}")
    print()

    # Naive Bayes
    print("--- Naive Bayes ---")
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Accuracy: {accuracy:.4f}")
    print()

    print("=== Algorithm Comparison Summary ===")
    print("Linear Regression: Best for continuous target prediction")
    print("Logistic Regression: Binary classification with probability outputs")
    print("KNN: Simple, instance-based learning")
    print("K-Means: Unsupervised clustering")
    print("Decision Tree: Interpretable classification rules")
    print("Naive Bayes: Fast probabilistic classification")


if __name__ == "__main__":
    demonstrate_ml_algorithms()
