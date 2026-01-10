import os
import unittest
import pandas as pd

# Dynamic Import Helper
import importlib.util


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pt_path = os.path.join(
    os.getcwd(), "UTILS - Strategies - Pairs Trading", "pairs_trading.py"
)
pt_module = load_module_from_path("pairs_trading", pt_path)


class TestPairsTrading(unittest.TestCase):

    def test_spread_calculation(self):
        x = pd.Series([100, 101, 102])
        y = pd.Series([105, 106, 107])
        # Spread = Y - X (ratio 1) => 5, 5, 5
        spread = pt_module.calculate_spread(x, y, 1.0)
        self.assertTrue((spread == 5).all())

    def test_zscore(self):
        # Create a series with known mean/std
        # 10, 10, 10 -> mean 10, std 0 -> zscore nan/inf usually, let's use varying
        s = pd.Series([10, 12, 14, 16, 18])
        z = pt_module.calculate_zscore(s, window=5)
        # Last element should be (18 - 14) / 3.16... ~ 1.26
        # Just check it returns a series of same shape
        self.assertEqual(len(z), 5)

    def test_signal_logic(self):
        # If Z > 2, signal should be -1
        z = pd.Series([0, 1, 2.5, 0.1])
        signals = pt_module.get_signal(z, entry_thresh=2.0, exit_thresh=0.5)

        # 0 -> 0
        # 1 -> 0 (no entry yet)
        # 2.5 -> -1 (entry short)
        # 0.1 -> 0 (exit)
        self.assertEqual(signals.iloc[2], -1)
        self.assertEqual(signals.iloc[3], 0)


if __name__ == "__main__":
    unittest.main()
