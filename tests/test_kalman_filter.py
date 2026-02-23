import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import module
# Note: Since the folder has spaces, we can't do a standard import easily without helper
# We will use importlib for these specific paths
import importlib.util


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kf_path = os.path.join(os.getcwd(), "UTILS - Quantitative Methods - Kalman Filter", "kalman_filter.py")
kf_module = load_module_from_path("kalman_filter", kf_path)


class TestKalmanFilter(unittest.TestCase):
    def test_initialization(self):
        kf = kf_module.KalmanFilter1D(process_variance=0.1, measurement_variance=0.1, estimated_value=10)
        self.assertEqual(kf.estimated_value, 10)

    def test_convergence(self):
        # If we feed it the same value repeatedly, it should converge near that value
        target = 100.0
        kf = kf_module.KalmanFilter1D(process_variance=1e-5, measurement_variance=0.1, estimated_value=50.0)

        for _ in range(50):
            kf.predict()
            kf.update(target)

        # Should be close to 100
        self.assertAlmostEqual(kf.estimated_value, target, delta=1.0)


if __name__ == "__main__":
    unittest.main()
