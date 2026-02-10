import unittest
import sys
import os

import unittest
import sys
import os
import importlib.util

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Resolve paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
kelly_path = os.path.join(base_path, "UTILS - Kelly Criterion", "kelly_criterion.py")
risk_parity_path = os.path.join(base_path, "UTILS - Risk Parity", "risk_parity.py")

# Load modules
kelly_module = load_module_from_path("kelly_criterion", kelly_path)
risk_parity_module = load_module_from_path("risk_parity", risk_parity_path)

class TestQuantUtils(unittest.TestCase):
    def test_kelly_criterion(self):
        # p=0.6, b=1 -> f = (1*0.6 - 0.4)/1 = 0.2
        self.assertAlmostEqual(kelly_module.calculate_kelly_fraction(0.6, 1.0), 0.2)
        # p=0.35, b=3 -> f = (3*0.35 - 0.65)/3 = (1.05 - 0.65)/3 = 0.4/3 = 0.1333
        self.assertAlmostEqual(kelly_module.calculate_kelly_fraction(0.35, 3.0), 0.4/3.0)
        # Losing strategy should return 0
        self.assertEqual(kelly_module.calculate_kelly_fraction(0.4, 1.0), 0.0)

    def test_risk_parity_weights(self):
        vols = [0.2, 0.1]
        weights = risk_parity_module.calculate_inverse_vol_weights(vols)
        # Inverse vols: 5, 10 -> Sum=15 -> Weights: 5/15, 10/15 = 1/3, 2/3
        self.assertAlmostEqual(weights[0], 1/3.0)
        self.assertAlmostEqual(weights[1], 2/3.0)
        self.assertAlmostEqual(sum(weights), 1.0)


if __name__ == "__main__":
    unittest.main()
