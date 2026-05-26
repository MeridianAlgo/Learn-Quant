# Dynamic Import Helper
# Note: We test the pure calculation functions only (simulate_gbm_path,
# price_call_option_mc) without spawning actual process pools, which keeps
# tests fast and avoids Windows process-spawning overhead in CI.
import importlib.util
import os
import unittest

import numpy as np


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mp_path = os.path.join(os.getcwd(), "UTILS - Advanced Python - Multiprocessing", "multiprocessing_tutorial.py")
mp = load_module_from_path("multiprocessing_tutorial", mp_path)


class TestGBMSimulation(unittest.TestCase):
    def test_gbm_returns_positive_price(self):
        """simulate_gbm_path should always return a positive terminal price."""
        args = (0, 100.0, 0.08, 0.20, 1.0, 252)
        result = mp.simulate_gbm_path(args)
        self.assertGreater(result, 0.0)

    def test_gbm_reproducible_with_seed(self):
        """Same seed should produce identical terminal price."""
        args = (42, 100.0, 0.08, 0.20, 1.0, 252)
        price_a = mp.simulate_gbm_path(args)
        price_b = mp.simulate_gbm_path(args)
        self.assertAlmostEqual(price_a, price_b, places=10)

    def test_gbm_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different terminal prices."""
        price_1 = mp.simulate_gbm_path((1, 100.0, 0.08, 0.20, 1.0, 252))
        price_2 = mp.simulate_gbm_path((2, 100.0, 0.08, 0.20, 1.0, 252))
        self.assertNotAlmostEqual(price_1, price_2, places=4)

    def test_gbm_zero_vol_equals_drift(self):
        """With sigma=0, terminal price should equal S0 * exp(mu * T)."""
        S0, mu, T = 100.0, 0.10, 1.0
        args = (0, S0, mu, 0.0, T, 252)
        result = mp.simulate_gbm_path(args)
        expected = S0 * np.exp(mu * T)
        self.assertAlmostEqual(result, expected, delta=0.01)

    def test_gbm_scales_with_s0(self):
        """Doubling S0 should double the terminal price (same proportional path)."""
        args_100 = (5, 100.0, 0.05, 0.20, 1.0, 252)
        args_200 = (5, 200.0, 0.05, 0.20, 1.0, 252)
        price_100 = mp.simulate_gbm_path(args_100)
        price_200 = mp.simulate_gbm_path(args_200)
        # Same seed → same random path → ratio should equal 2
        self.assertAlmostEqual(price_200 / price_100, 2.0, places=10)

    def test_sequential_returns_correct_count(self):
        """run_sequential should return exactly n_simulations terminal prices."""
        prices = mp.run_sequential(20, 100.0, 0.08, 0.20, 1.0, 252)
        self.assertEqual(len(prices), 20)
        self.assertTrue(all(p > 0 for p in prices))

    def test_option_pricer_positive_price(self):
        """price_call_option_mc should return a positive call price."""
        args = (0.20, 100.0, 100.0, 0.05, 1.0, 500, 0)
        result = mp.price_call_option_mc(args)
        self.assertIn("sigma", result)
        self.assertIn("call_price", result)
        self.assertGreater(result["call_price"], 0.0)

    def test_higher_vol_higher_call_price(self):
        """A higher volatility should produce a higher call price (core Black-Scholes result)."""
        low_vol_result = mp.price_call_option_mc((0.10, 100.0, 100.0, 0.05, 1.0, 2000, 1))
        high_vol_result = mp.price_call_option_mc((0.40, 100.0, 100.0, 0.05, 1.0, 2000, 2))
        self.assertGreater(high_vol_result["call_price"], low_vol_result["call_price"])

    def test_deep_itm_call_price_positive(self):
        """A deeply in-the-money call (S0 >> K) should have a large positive price."""
        # S0=200, K=100 → deeply in the money → call price ≈ 200 * exp(-r*T) - 100 * exp(-r*T)
        result = mp.price_call_option_mc((0.20, 200.0, 100.0, 0.05, 1.0, 2000, 0))
        self.assertGreater(result["call_price"], 50.0)


if __name__ == "__main__":
    unittest.main()
