import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

class StatisticalArbitrage:
    """
    A basic Statistical Arbitrage strategy focusing on pairs trading based on cointegration.
    """
    def __init__(self, entry_zscore=2.0, exit_zscore=0.0):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore

    def test_cointegration(self, series1: pd.Series, series2: pd.Series):
        """
        Test if two series are cointegrated.
        """
        score, p_value, _ = coint(series1, series2)
        print(f"Cointegration test p-value: {p_value:.4f}")
        return p_value < 0.05

    def calculate_spread(self, series1: pd.Series, series2: pd.Series):
        """
        Calculate the spread between two cointegrated series.
        Using a simple price ratio for demonstration. 
        In practice, use hedge ratio from regression.
        """
        # Simple log spread: log(P1) - log(P2)
        spread = np.log(series1) - np.log(series2)
        return spread

    def generate_signals(self, spread: pd.Series, window=30):
        """
        Generate trading signals based on z-score of the spread.
        """
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        zscore = (spread - mean) / std

        signals = pd.DataFrame(index=spread.index)
        signals['zscore'] = zscore
        signals['long_entry'] = zscore < -self.entry_zscore
        signals['short_entry'] = zscore > self.entry_zscore
        signals['exit'] = abs(zscore) <= self.exit_zscore

        return signals

if __name__ == "__main__":
    np.random.seed(42)
    # Generate random walk
    prices1 = np.random.randn(200).cumsum() + 100
    # Generate cointegrated series
    prices2 = prices1 + np.random.randn(200) * 0.5
    
    df1 = pd.Series(prices1)
    df2 = pd.Series(prices2)
    
    arb = StatisticalArbitrage()
    is_coint = arb.test_cointegration(df1, df2)
    
    if is_coint:
        spread = arb.calculate_spread(df1, df2)
        signals = arb.generate_signals(spread)
        print("Generated Signals Summary:")
        print(signals.sum())
