import unittest
import numpy as np
import os
import sys
import unittest
import importlib.util
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Define base folder path
perf_folder = os.path.join(os.getcwd(), "UTILS - Quantitative Methods - Performance Analysis")

# Load modules
hurst_module = load_module_from_path("hurst_exponent", os.path.join(perf_folder, "hurst_exponent.py"))
omega_module = load_module_from_path("omega_ratio", os.path.join(perf_folder, "omega_ratio.py"))
tail_module = load_module_from_path("tail_ratio", os.path.join(perf_folder, "tail_ratio.py"))
gp_module = load_module_from_path("gain_to_pain_ratio", os.path.join(perf_folder, "gain_to_pain_ratio.py"))
active_module = load_module_from_path("active_performance", os.path.join(perf_folder, "active_performance.py"))

class TestPerformanceAnalysis(unittest.TestCase):
    def setUp(self):
        # Setting seed to ensure deterministic tests
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.01, 252)
        self.benchmark = np.random.normal(0.0005, 0.01, 252)
        self.alpha_returns = self.benchmark + 0.0001 + np.random.normal(0, 0.002, 252)

    def test_hurst_exponent(self):
        # A simple trending series should have H > 0.5
        trending = np.cumsum(np.random.randn(1000) + 0.5)
        h = hurst_module.compute_hurst(trending)
        # Check if it yields a float and is roughly in expected range
        self.assertIsInstance(h, float)
        self.assertGreater(h, 0.3)

    def test_omega_ratio(self):
        # Test with target 0%
        o_ratio = omega_module.omega_ratio(self.returns, target=0)
        self.assertIsInstance(o_ratio, float)
        self.assertGreater(o_ratio, 0)
        
        # Test if infinite with no losses
        no_losses = np.array([0.01, 0.02, 0.03])
        self.assertEqual(omega_module.omega_ratio(no_losses), np.inf)

    def test_tail_ratio(self):
        # Test basic functionality
        ratio = tail_module.compute_tail_ratio(self.returns)
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 0)

    def test_gain_to_pain(self):
        # Test basic functionality
        gp_ratio = gp_module.gain_to_pain_ratio(self.returns)
        self.assertIsInstance(gp_ratio, float)
        # Should be > 0 for positive returns
        self.assertGreaterEqual(gp_ratio, 0)

    def test_active_metrics(self):
        # Test if returns dictionary with correct keys
        results = active_module.active_metrics(self.alpha_returns, self.benchmark)
        self.assertIn('tracking_error', results)
        self.assertIn('information_ratio', results)
        self.assertGreater(results['tracking_error'], 0)
        
        # Test error for mismatched lengths
        with self.assertRaises(ValueError):
            active_module.active_metrics([0.01], [0.01, 0.02])


if __name__ == '__main__':
    unittest.main()
