"""Test suite for Quantitative Methods - Regression Analysis utility."""

import unittest
import numpy as np
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_simple_linear_regression(self):
        """Test simple linear regression (y = mx + c)."""
        x = np.array([1, 2, 3, 4, 5])
        # y = 2x + 1
        y = np.array([3, 5, 7, 9, 11])

        coeffs = np.polyfit(x, y, deg=1)
        slope = coeffs[0]
        intercept = coeffs[1]

        self.assertAlmostEqual(slope, 2.0, places=5)
        self.assertAlmostEqual(intercept, 1.0, places=5)

    def test_beta_calculation(self):
        """Test beta calculation logic."""
        market_returns = np.random.normal(0, 0.01, 100)
        # Stock perfectly correlated with market, beta = 1.5
        stock_returns = 1.5 * market_returns

        coeffs = np.polyfit(market_returns, stock_returns, deg=1)
        beta = coeffs[0]

        self.assertAlmostEqual(beta, 1.5, places=5)

    def test_r_squared(self):
        """Test R-squared calculation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])  # Perfect fit

        coeffs = np.polyfit(x, y, deg=1)
        y_pred = coeffs[1] + coeffs[0] * x

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        self.assertAlmostEqual(r_squared, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
