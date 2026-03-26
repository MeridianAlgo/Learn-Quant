# Dynamic Import Helper
import importlib.util
import os
import unittest

import numpy as np
import pandas as pd


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mr_path = os.path.join(
    os.getcwd(), "UTILS - Strategies - Mean Reversion", "mean_reversion_strategy.py"
)
mr = load_module_from_path("mean_reversion_strategy", mr_path)


class TestMeanReversionStrategy(unittest.TestCase):
    def test_price_series_length(self):
        """Generated price series should have exactly n_points elements."""
        prices = mr.generate_mean_reverting_prices(n_points=100, seed=0)
        self.assertEqual(len(prices), 100)

    def test_prices_are_positive(self):
        """Ornstein-Uhlenbeck prices should stay positive for reasonable parameters."""
        prices = mr.generate_mean_reverting_prices(n_points=300, seed=42)
        self.assertTrue((prices > 0).all(), "All prices should be positive")

    def test_prices_mean_revert(self):
        """The mean of the generated series should be close to theta=100."""
        prices = mr.generate_mean_reverting_prices(n_points=1000, seed=7)
        # With 1000 observations and kappa=0.05, the mean should be within 5 of theta=100
        self.assertAlmostEqual(prices.mean(), 100.0, delta=5.0)

    def test_bollinger_bands_shape(self):
        """Bollinger Bands DataFrame should have 4 columns and same length as input."""
        prices = mr.generate_mean_reverting_prices(n_points=100)
        bands = mr.calculate_bollinger_bands(prices, window=20, num_std=2.0)
        self.assertEqual(bands.shape, (100, 4))
        self.assertIn("SMA", bands.columns)
        self.assertIn("Upper", bands.columns)
        self.assertIn("Lower", bands.columns)
        self.assertIn("Z_Score", bands.columns)

    def test_upper_above_lower(self):
        """Upper band must always be above the lower band (where not NaN)."""
        prices = mr.generate_mean_reverting_prices(n_points=100)
        bands = mr.calculate_bollinger_bands(prices, window=20, num_std=2.0)
        valid = bands.dropna()
        self.assertTrue((valid["Upper"] > valid["Lower"]).all())

    def test_rsi_bounds(self):
        """RSI values must be in [0, 100]."""
        prices = mr.generate_mean_reverting_prices(n_points=200)
        rsi = mr.calculate_rsi(prices, period=14)
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

    def test_rsi_length(self):
        """RSI series must have same length as input price series."""
        prices = mr.generate_mean_reverting_prices(n_points=50)
        rsi = mr.calculate_rsi(prices, period=14)
        self.assertEqual(len(rsi), len(prices))

    def test_signal_values(self):
        """Signals should only contain -1, 0, or +1."""
        prices = mr.generate_mean_reverting_prices(n_points=200)
        bands = mr.calculate_bollinger_bands(prices, window=20)
        rsi = mr.calculate_rsi(prices, period=14)
        signals = mr.generate_signals(prices, bands, rsi)
        unique_vals = set(signals.dropna().unique())
        self.assertTrue(unique_vals.issubset({-1.0, 0.0, 1.0}))

    def test_backtest_output_columns(self):
        """Backtest DataFrame should contain the expected output columns."""
        prices = mr.generate_mean_reverting_prices(n_points=100)
        signals = pd.Series(np.zeros(100))
        results = mr.backtest(prices, signals)
        for col in ["Price", "Signal", "Cumulative_Market", "Cumulative_Strategy"]:
            self.assertIn(col, results.columns)

    def test_flat_signal_equals_buy_hold(self):
        """A strategy with all-zero signals should match the buy-and-hold cumulative return."""
        prices = mr.generate_mean_reverting_prices(n_points=100, seed=1)
        flat_signals = pd.Series(np.zeros(100))
        results = mr.backtest(prices, flat_signals)
        # All-zero signals mean zero strategy returns every day
        self.assertAlmostEqual(results["Cumulative_Strategy"].iloc[-1], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
