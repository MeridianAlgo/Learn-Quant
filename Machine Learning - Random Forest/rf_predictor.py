import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class RandomForestPredictor:
    """
    A Random Forest model for predicting time series returns.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the random forest model.
        """
        self.model.fit(X, y)
        print("Model trained.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict returns using the trained model.
        """
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Evaluate the model's performance.
        """
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    # Generate some dummy data
    X = pd.DataFrame(np.random.randn(100, 5), columns=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"])
    y = pd.Series(np.random.randn(100))

    predictor = RandomForestPredictor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictor.train(X_train, y_train)
    predictor.evaluate(X_test, y_test)
