# Dynamic Import Helper
import importlib.util
import os
import unittest


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ps_path = os.path.join(os.getcwd(), "UTILS - Finance - Position Sizing", "position_sizing_tutorial.py")
ps = load_module_from_path("position_sizing_tutorial", ps_path)


class TestFixedFractional(unittest.TestCase):
    def test_dollar_risk_calculation(self):
        """Dollar risk = portfolio_value × risk_per_trade_pct."""
        result = ps.fixed_fractional_sizing(100_000, 0.01, 0.05)
        self.assertAlmostEqual(result["dollar_risk"], 1000.0, places=2)

    def test_position_value_calculation(self):
        """Position value = dollar_risk / stop_loss_pct."""
        result = ps.fixed_fractional_sizing(100_000, 0.01, 0.05)
        self.assertAlmostEqual(result["position_value"], 20_000.0, places=2)

    def test_position_pct_of_portfolio(self):
        """Position as fraction of portfolio should equal position_value / portfolio."""
        result = ps.fixed_fractional_sizing(100_000, 0.01, 0.05)
        expected_pct = 20_000.0 / 100_000.0
        self.assertAlmostEqual(result["position_pct_of_portfolio"], expected_pct, places=5)

    def test_tighter_stop_smaller_position(self):
        """A tighter stop-loss should allow a smaller position for the same dollar risk."""
        wide_stop = ps.fixed_fractional_sizing(100_000, 0.01, 0.05)  # 5% stop
        tight_stop = ps.fixed_fractional_sizing(100_000, 0.01, 0.10)  # 10% stop
        self.assertGreater(wide_stop["position_value"], tight_stop["position_value"])


class TestKellyCriterion(unittest.TestCase):
    def test_known_kelly_value(self):
        """Kelly fraction for p=0.55, b=1.5 should be approximately 0.18."""
        # f* = p - q/b = 0.55 - 0.45/1.5 = 0.55 - 0.30 = 0.25
        kelly_f = ps.kelly_criterion(0.55, 1.5)
        self.assertAlmostEqual(kelly_f, 0.25, places=5)

    def test_no_edge_returns_zero(self):
        """A strategy with no edge (expected value <= 0) should return 0."""
        # p=0.4, b=1.0 → f* = 0.4 - 0.6/1.0 = -0.2 → clipped to 0
        kelly_f = ps.kelly_criterion(0.40, 1.0)
        self.assertEqual(kelly_f, 0.0)

    def test_high_win_rate_high_kelly(self):
        """A strategy with 70% win rate and 2x odds should have a large Kelly fraction."""
        kelly_f = ps.kelly_criterion(0.70, 2.0)
        # f* = 0.70 - 0.30/2.0 = 0.70 - 0.15 = 0.55
        self.assertAlmostEqual(kelly_f, 0.55, places=5)

    def test_kelly_in_valid_range(self):
        """Kelly fraction should always be between 0 and 1 for reasonable inputs."""
        for p in [0.4, 0.5, 0.55, 0.6, 0.7]:
            for b in [1.0, 1.5, 2.0, 3.0]:
                kelly_f = ps.kelly_criterion(p, b)
                self.assertGreaterEqual(kelly_f, 0.0)
                self.assertLessEqual(kelly_f, 1.0)

    def test_kelly_simulation_length(self):
        """Growth simulation DataFrame should have n_trades + 1 rows."""
        kelly_f = ps.kelly_criterion(0.55, 1.5)
        df = ps.kelly_growth_simulation(kelly_f, 0.55, 1.5, n_trades=100, seed=0)
        self.assertEqual(len(df), 101)  # row 0 is starting value

    def test_kelly_simulation_starts_at_one(self):
        """All strategies in the simulation should start at portfolio = $1."""
        kelly_f = ps.kelly_criterion(0.55, 1.5)
        df = ps.kelly_growth_simulation(kelly_f, 0.55, 1.5, n_trades=50, seed=1)
        for col in ["Full Kelly", "Half Kelly", "Quarter Kelly", "Fixed 1%"]:
            self.assertAlmostEqual(df[col].iloc[0], 1.0, places=10)

    def test_full_kelly_exceeds_half_kelly_eventually(self):
        """Full Kelly should grow faster than Half Kelly over a long enough horizon (positive edge)."""
        kelly_f = ps.kelly_criterion(0.60, 2.0)
        df = ps.kelly_growth_simulation(kelly_f, 0.60, 2.0, n_trades=1000, seed=42)
        # Over 1000 trades with strong edge, Full Kelly should end higher
        self.assertGreater(df["Full Kelly"].iloc[-1], df["Half Kelly"].iloc[-1])


class TestVolatilityTargeting(unittest.TestCase):
    def test_notional_formula(self):
        """Notional should equal portfolio × (target_vol / asset_vol)."""
        result = ps.volatility_targeting(0.10, 0.25, 500_000, 100.0)
        expected_notional = 500_000 * (0.10 / 0.25)
        self.assertAlmostEqual(result["notional_exposure"], expected_notional, places=2)

    def test_lower_asset_vol_larger_position(self):
        """Lower asset volatility → larger position needed to hit target vol."""
        low_vol = ps.volatility_targeting(0.10, 0.15, 100_000, 50.0)
        high_vol = ps.volatility_targeting(0.10, 0.30, 100_000, 50.0)
        self.assertGreater(low_vol["notional_exposure"], high_vol["notional_exposure"])

    def test_shares_non_negative(self):
        """Number of shares to buy should never be negative."""
        result = ps.volatility_targeting(0.10, 0.20, 100_000, 50.0)
        self.assertGreaterEqual(result["n_shares"], 0)

    def test_actual_vol_close_to_target(self):
        """Actual portfolio vol after rounding should be close (within 1%) of the target."""
        result = ps.volatility_targeting(0.10, 0.20, 1_000_000, 10.0)
        self.assertAlmostEqual(result["actual_portfolio_vol"], 0.10, delta=0.01)


class TestRiskOfRuin(unittest.TestCase):
    def test_returns_probability(self):
        """Risk of ruin should return a value between 0 and 1."""
        ror = ps.risk_of_ruin(0.55, 1.5, 0.02, n_simulations=100, n_trades=100, seed=0)
        self.assertGreaterEqual(ror, 0.0)
        self.assertLessEqual(ror, 1.0)

    def test_high_risk_higher_ruin(self):
        """Larger risk per trade should produce higher Risk of Ruin."""
        ror_small = ps.risk_of_ruin(0.55, 1.5, 0.01, n_simulations=500, n_trades=500, seed=7)
        ror_large = ps.risk_of_ruin(0.55, 1.5, 0.10, n_simulations=500, n_trades=500, seed=7)
        self.assertGreater(ror_large, ror_small)

    def test_no_edge_nonzero_ruin(self):
        """A strategy with no edge should have meaningfully positive Risk of Ruin."""
        ror = ps.risk_of_ruin(0.45, 1.0, 0.05, n_simulations=500, n_trades=500, seed=3)
        self.assertGreater(ror, 0.0)


if __name__ == "__main__":
    unittest.main()
