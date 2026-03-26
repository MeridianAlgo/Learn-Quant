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


fm_path = os.path.join(os.getcwd(), "UTILS - Quantitative Methods - Factor Models", "factor_models_tutorial.py")
fm = load_module_from_path("factor_models_tutorial", fm_path)


class TestFactorModels(unittest.TestCase):
    def test_factor_data_shape(self):
        """Factor data should have n_periods rows and the expected 4 columns."""
        df = fm.generate_synthetic_factor_data(n_periods=60, seed=0)
        self.assertEqual(df.shape[0], 60)
        for col in ["MKT_RF", "SMB", "HML", "RF"]:
            self.assertIn(col, df.columns)

    def test_factor_data_means(self):
        """Factor means should be close to the specified theoretical means."""
        df = fm.generate_synthetic_factor_data(n_periods=5000, seed=1)
        # MKT mean ≈ 0.5%/month, SMB ≈ 0.2%, HML ≈ 0.3%
        self.assertAlmostEqual(df["MKT_RF"].mean(), 0.5, delta=0.3)
        self.assertAlmostEqual(df["SMB"].mean(), 0.2, delta=0.3)
        self.assertAlmostEqual(df["HML"].mean(), 0.3, delta=0.3)

    def test_stock_returns_length(self):
        """Generated stock returns should have the same length as factor data."""
        factors = fm.generate_synthetic_factor_data(n_periods=120)
        returns = fm.generate_stock_returns(
            factors, alpha=0.05, beta_mkt=1.0, beta_smb=0.0, beta_hml=0.0, idio_vol=2.0, seed=42
        )
        self.assertEqual(len(returns), 120)

    def test_zero_idio_returns_deterministic(self):
        """With idio_vol=0, returns should be fully determined by factors."""
        factors = fm.generate_synthetic_factor_data(n_periods=50, seed=3)
        returns_a = fm.generate_stock_returns(
            factors, alpha=0.1, beta_mkt=1.0, beta_smb=0.5, beta_hml=0.3, idio_vol=0.0, seed=99
        )
        returns_b = fm.generate_stock_returns(
            factors, alpha=0.1, beta_mkt=1.0, beta_smb=0.5, beta_hml=0.3, idio_vol=0.0, seed=0
        )
        # With no idiosyncratic noise, returns are deterministic regardless of seed
        pd.testing.assert_series_equal(returns_a, returns_b)

    def test_ols_recovers_known_coefficients(self):
        """OLS should approximately recover the true beta values with enough data."""
        np.random.seed(0)
        n = 500
        # Create a synthetic X matrix with const and one factor
        X = pd.DataFrame({"const": np.ones(n), "factor": np.random.randn(n)})
        true_alpha = 0.10
        true_beta = 1.50
        y = pd.Series(true_alpha * X["const"] + true_beta * X["factor"] + np.random.randn(n) * 0.1)

        result = fm.run_ols_regression(y, X)
        coefs = result["coefficients"]

        self.assertAlmostEqual(coefs["const"], true_alpha, delta=0.05)
        self.assertAlmostEqual(coefs["factor"], true_beta, delta=0.05)

    def test_ols_r_squared_high_signal(self):
        """With very low noise, R-squared should be close to 1.0."""
        np.random.seed(7)
        n = 200
        X = pd.DataFrame({"const": np.ones(n), "f": np.random.randn(n)})
        y = pd.Series(0.05 + 1.2 * X["f"] + np.random.randn(n) * 0.001)
        result = fm.run_ols_regression(y, X)
        self.assertGreater(result["r_squared"], 0.99)

    def test_ols_r_squared_pure_noise(self):
        """With pure noise (no real factor), R-squared should be near 0."""
        np.random.seed(8)
        n = 200
        X = pd.DataFrame({"const": np.ones(n), "f": np.random.randn(n)})
        y = pd.Series(np.random.randn(n))  # no factor relationship
        result = fm.run_ols_regression(y, X)
        self.assertLess(result["r_squared"], 0.10)

    def test_ols_output_keys(self):
        """OLS result dict should contain all required keys."""
        np.random.seed(1)
        n = 100
        X = pd.DataFrame({"const": np.ones(n), "MKT_RF": np.random.randn(n)})
        y = pd.Series(np.random.randn(n))
        result = fm.run_ols_regression(y, X)
        for key in ["coefficients", "t_stats", "r_squared", "residual_std", "n_obs"]:
            self.assertIn(key, result)

    def test_factor_contribution_sums(self):
        """Factor contributions should sum to approximately the average excess return."""
        factors = fm.generate_synthetic_factor_data(n_periods=200, seed=5)
        returns = fm.generate_stock_returns(
            factors, alpha=0.05, beta_mkt=1.0, beta_smb=0.3, beta_hml=0.2, idio_vol=1.0, seed=10
        )
        X = factors[["MKT_RF", "SMB", "HML"]].copy()
        X.insert(0, "const", 1.0)
        result = fm.run_ols_regression(returns, X)
        contrib_df = fm.factor_contribution(result, factors)
        total_explained = contrib_df["Contribution_%"].sum()
        # Total explained return should be close to the mean return
        self.assertAlmostEqual(total_explained, returns.mean(), delta=0.5)


if __name__ == "__main__":
    unittest.main()
