"""Test suite for Quantitative Methods - Stochastic Processes utility."""

import os
import sys
import unittest

import numpy as np

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestStochastic(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_brownian_motion_properties(self):
        """Test properties of Brownian Motion."""
        T = 1.0
        N = 1000
        dt = T / N
        n_sims = 100

        # dW ~ N(0, dt)
        dW = np.random.normal(0, np.sqrt(dt), (n_sims, N))
        W = np.cumsum(dW, axis=1)

        # Mean should be close to 0
        final_values = W[:, -1]
        mean_final = np.mean(final_values)

        # Standard error of mean is std/sqrt(n) -> 1/10 = 0.1
        # So mean should be within ~2-3 std errors (0.3)
        self.assertTrue(abs(mean_final) < 0.3)

        # Variance should be close to T (1.0)
        var_final = np.var(final_values)
        self.assertTrue(abs(var_final - 1.0) < 0.3)

    def test_gbm_positivity(self):
        """Test that Geometric Brownian Motion stays positive."""
        S0 = 100
        mu = 0.05
        sigma = 0.2
        T = 1.0
        N = 100
        dt = T / N

        # Simulate one path
        Z = np.random.normal(0, 1, N)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        returns = drift + diffusion

        price_path = S0 * np.exp(np.cumsum(returns))

        # All prices should be positive
        self.assertTrue(np.all(price_path > 0))


if __name__ == "__main__":
    unittest.main()
