"""Test suite for Quantitative Methods - Optimization utility."""

import unittest
import numpy as np
from scipy.optimize import minimize
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestOptimization(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_basic_minimization(self):
        """Test simple function minimization."""

        def objective(x):
            return (x[0] - 3) ** 2 + 5

        result = minimize(objective, [0])

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x[0], 3.0, places=4)
        self.assertAlmostEqual(result.fun, 5.0, places=4)

    def test_portfolio_optimization(self):
        """Test portfolio variance minimization."""
        # Simple 2-asset case
        # Asset A: 10% vol, Asset B: 10% vol, Correlation: 0
        # Equal weight should reduce vol to 10% / sqrt(2) ≈ 7.07%

        cov_matrix = np.array([[0.01, 0.00], [0.00, 0.01]])

        def volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = ((0, 1), (0, 1))

        result = minimize(
            volatility,
            [0.5, 0.5],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x[0], 0.5, places=3)
        self.assertAlmostEqual(result.x[1], 0.5, places=3)
        self.assertAlmostEqual(result.fun, np.sqrt(0.005), places=4)


if __name__ == "__main__":
    unittest.main()
