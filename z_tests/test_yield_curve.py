# Dynamic Import Helper
import importlib.util
import os
import unittest

import numpy as np


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


yc_path = os.path.join(os.getcwd(), "UTILS - Finance - Yield Curve", "yield_curve_tutorial.py")
yc = load_module_from_path("yield_curve_tutorial", yc_path)


class TestYieldCurve(unittest.TestCase):
    def test_sample_data_shape(self):
        """Sample treasury yield DataFrame should have 11 rows and 2 columns."""
        df = yc.get_sample_treasury_yields()
        self.assertEqual(len(df), 11)
        self.assertIn("maturity", df.columns)
        self.assertIn("yield_pct", df.columns)

    def test_sample_data_positive_yields(self):
        """All sample par yields must be positive."""
        df = yc.get_sample_treasury_yields()
        self.assertTrue((df["yield_pct"] > 0).all())

    def test_sample_data_positive_maturities(self):
        """All maturities must be strictly positive."""
        df = yc.get_sample_treasury_yields()
        self.assertTrue((df["maturity"] > 0).all())

    def test_nelson_siegel_output_length(self):
        """Nelson-Siegel should return an array of the same length as input maturities."""
        mats = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
        result = yc.nelson_siegel(mats, beta0=4.0, beta1=-1.0, beta2=1.5, tau=2.0)
        self.assertEqual(len(result), len(mats))

    def test_nelson_siegel_long_rate_converges(self):
        """At very long maturities, Nelson-Siegel yield should approach beta0."""
        beta0 = 4.5
        very_long = np.array([1000.0])  # effectively infinity
        result = yc.nelson_siegel(very_long, beta0=beta0, beta1=-1.0, beta2=1.5, tau=2.0)
        self.assertAlmostEqual(result[0], beta0, delta=0.1)

    def test_nelson_siegel_all_positive(self):
        """Fitted yields should be positive for realistic parameter values."""
        mats = np.linspace(0.25, 30, 50)
        result = yc.nelson_siegel(mats, beta0=4.5, beta1=-0.5, beta2=1.0, tau=2.0)
        self.assertTrue((result > 0).all())

    def test_fit_returns_all_keys(self):
        """Fitted parameters dict should contain beta0, beta1, beta2, tau, and rmse."""
        df = yc.get_sample_treasury_yields()
        params = yc.fit_nelson_siegel(df["maturity"].values, df["yield_pct"].values)
        for key in ["beta0", "beta1", "beta2", "tau", "rmse"]:
            self.assertIn(key, params)

    def test_fit_rmse_reasonable(self):
        """RMSE of the Nelson-Siegel fit to sample data should be below 0.5%."""
        df = yc.get_sample_treasury_yields()
        params = yc.fit_nelson_siegel(df["maturity"].values, df["yield_pct"].values)
        self.assertLess(params["rmse"], 0.5)

    def test_forward_rates_shape(self):
        """compute_forward_rates should return DataFrame with 3 columns."""
        mats = np.array([1.0, 2.0, 5.0, 10.0])
        spots = np.array([4.0, 4.2, 4.5, 4.8])
        fwd = yc.compute_forward_rates(mats, spots)
        self.assertEqual(len(fwd), 4)
        for col in ["Maturity_Yrs", "Spot_Rate_Pct", "Forward_Rate_Pct"]:
            self.assertIn(col, fwd.columns)

    def test_forward_rate_first_equals_spot(self):
        """The first forward rate should equal the first spot rate."""
        mats = np.array([1.0, 2.0, 5.0])
        spots = np.array([4.0, 4.5, 5.0])
        fwd = yc.compute_forward_rates(mats, spots)
        # First forward rate = first spot rate (no prior period to compute a forward from)
        self.assertAlmostEqual(fwd["Forward_Rate_Pct"].iloc[0], spots[0], places=3)

    def test_classify_normal_curve(self):
        """A curve with long > short by more than 0.3% should be classified as normal."""
        shape = yc.classify_curve_shape(short_rate=3.0, long_rate=4.5, mid_rate=4.0)
        self.assertIn("Normal", shape)

    def test_classify_inverted_curve(self):
        """A curve with short > long by more than 0.1% should be classified as inverted."""
        shape = yc.classify_curve_shape(short_rate=5.0, long_rate=4.5, mid_rate=4.8)
        self.assertIn("Inverted", shape)


if __name__ == "__main__":
    unittest.main()
