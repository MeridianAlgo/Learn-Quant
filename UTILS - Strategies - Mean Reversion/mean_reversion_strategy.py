"""Mean Reversion Strategy Simulator.

Run with:
    python mean_reversion_strategy.py

Mean reversion is one of the oldest ideas in quantitative finance: prices that
deviate far from their historical average tend to "snap back" toward it.  This
script implements a Bollinger Band + RSI mean reversion strategy step by step,
showing how to generate signals, manage positions, and measure performance.

Key Concepts:
- Mean Reversion: Assets oscillate around a long-run equilibrium price.
- Bollinger Bands: Upper/lower bands at (SMA ± k*std) capturing ~95% of price action.
- Z-Score: Standardised distance of price from its rolling mean (how "extreme" is today?).
- RSI Filter: Relative Strength Index used to confirm oversold/overbought conditions.
- Ornstein-Uhlenbeck Process: The continuous-time model that generates mean-reverting prices.
"""

import numpy as np
import pandas as pd

# ─── 1. SYNTHETIC DATA ───────────────────────────────────────────────────────

def generate_mean_reverting_prices(n_points: int = 300, seed: int = 42) -> pd.Series:
    """
    Generate a synthetic price series that exhibits mean-reverting behaviour.

    We use an Ornstein-Uhlenbeck (OU) process — the continuous-time analog of
    an AR(1) model — to simulate prices pulled back toward a long-run mean theta.

    The discrete update rule is:
        x[t] = x[t-1] + kappa * (theta - x[t-1]) * dt + sigma * sqrt(dt) * eps

    where:
    - theta  = long-run mean (the "fair value" the price gravitates toward)
    - kappa  = mean-reversion speed (higher = snaps back faster)
    - sigma  = diffusion / noise level
    - eps    ~ N(0,1) white noise

    Args:
        n_points: Number of daily price observations to generate.
        seed:     Random seed for reproducibility across runs.

    Returns:
        pd.Series of simulated closing prices.
    """
    np.random.seed(seed)

    # OU process parameters — tuned to look like a realistic equity price
    theta = 100.0   # long-run mean: think of this as "fair value"
    kappa = 0.05    # mean-reversion speed (5% per day pulls it toward theta)
    sigma = 1.2     # noise level: controls daily price jitter
    dt = 1.0        # time step = 1 trading day

    prices = np.zeros(n_points)
    prices[0] = theta  # start at fair value

    for t in range(1, n_points):
        # Drift component: always pulling price back toward theta
        drift = kappa * (theta - prices[t - 1]) * dt
        # Random shock: adds day-to-day noise
        diffusion = sigma * np.sqrt(dt) * np.random.randn()
        prices[t] = prices[t - 1] + drift + diffusion

    return pd.Series(prices, name="Close")


# ─── 2. BOLLINGER BANDS ──────────────────────────────────────────────────────

def calculate_bollinger_bands(prices: pd.Series, window: int = 20,
                               num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands around a rolling mean.

    John Bollinger's indicator (1983) wraps a Simple Moving Average with
    bands set at ±k standard deviations.  Because ~95% of a normal distribution
    falls within ±2 std, touching the outer band is statistically unusual.

    Components:
    - Middle Band (SMA): rolling average — our "fair value" estimate each day.
    - Upper Band:        SMA + num_std * rolling_std → "statistically expensive".
    - Lower Band:        SMA - num_std * rolling_std → "statistically cheap".
    - Z-Score:          (price - SMA) / rolling_std — signed distance from mean.

    Args:
        prices:   Series of asset closing prices.
        window:   Rolling look-back period (20 days is the industry default).
        num_std:  Band width in standard deviations (2.0 covers ~95% of moves).

    Returns:
        DataFrame with columns: SMA, Upper, Lower, Z_Score.
    """
    # Rolling mean — our time-varying estimate of "fair value"
    sma = prices.rolling(window=window).mean()

    # Rolling standard deviation — measures how volatile recent prices have been
    rolling_std = prices.rolling(window=window).std()

    upper_band = sma + num_std * rolling_std
    lower_band = sma - num_std * rolling_std

    # Z-score: positive = above average (expensive), negative = below average (cheap)
    z_score = (prices - sma) / rolling_std

    return pd.DataFrame(
        {
            "SMA": sma,
            "Upper": upper_band,
            "Lower": lower_band,
            "Z_Score": z_score,
        }
    )


# ─── 3. RSI ──────────────────────────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) as a momentum confirmation filter.

    RSI measures the speed and magnitude of recent price changes on a 0–100 scale.
    - RSI > 70: Overbought — momentum confirms a Bollinger upper-band short signal.
    - RSI < 30: Oversold  — momentum confirms a Bollinger lower-band long signal.

    We use RSI as a *confirmation* tool alongside Bollinger Bands, not as a
    standalone signal, which significantly reduces false entries in trending markets.

    Formula:
        RSI = 100 – 100 / (1 + RS)
        RS  = EMA(Gains, period) / EMA(Losses, period)

    Wilder's original smoothing is replicated with pandas ewm(span=period, adjust=False).

    Args:
        prices: Series of asset closing prices.
        period: Look-back period (14 days is the original Wilder default).

    Returns:
        pd.Series of RSI values in the range [0, 100].
    """
    delta = prices.diff()

    # Separate daily price changes into gains and losses
    gains = delta.clip(lower=0)     # positive changes only; losses are zeroed
    losses = -delta.clip(upper=0)   # negative changes flipped positive; gains zeroed

    # Exponential smoothing of gains and losses (Wilder's method)
    avg_gain = gains.ewm(span=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, adjust=False).mean()

    # Relative Strength = ratio of average up-move to average down-move
    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("RSI")


# ─── 4. SIGNAL GENERATION ────────────────────────────────────────────────────

def generate_signals(prices: pd.Series, bands: pd.DataFrame, rsi: pd.Series,
                     rsi_oversold: float = 35.0, rsi_overbought: float = 65.0) -> pd.Series:
    """
    Generate mean-reversion trading signals using dual confirmation.

    Entry rules (BOTH Bollinger AND RSI must agree before entering):
    - LONG  (+1): price < Lower Band  AND RSI < rsi_oversold  → asset is cheap
    - SHORT (-1): price > Upper Band  AND RSI > rsi_overbought → asset is expensive

    Exit rules:
    - Exit LONG when price crosses back above the SMA (mean achieved, take profit).
    - Exit SHORT when price crosses back below the SMA (mean achieved, take profit).

    The dual-confirmation logic prevents entering during genuine trending markets
    where the Bollinger Band alone would generate many false signals.

    Args:
        prices:         Series of asset closing prices.
        bands:          DataFrame from calculate_bollinger_bands().
        rsi:            Series from calculate_rsi().
        rsi_oversold:   RSI level below which we confirm an oversold entry.
        rsi_overbought: RSI level above which we confirm an overbought entry.

    Returns:
        pd.Series of integer signals: +1 (long), –1 (short), 0 (flat).
    """
    signals = pd.Series(0.0, index=prices.index, name="Signal")
    position = 0  # tracks whether we are currently long (+1), short (-1), or flat (0)

    for i in range(len(prices)):
        # Skip the warm-up period while rolling indicators are still NaN
        if pd.isna(bands["Lower"].iloc[i]):
            continue

        price = prices.iloc[i]
        lower = bands["Lower"].iloc[i]
        upper = bands["Upper"].iloc[i]
        sma = bands["SMA"].iloc[i]
        curr_rsi = rsi.iloc[i]

        if position == 0:
            # No open position — scan for entry signals
            if price < lower and curr_rsi < rsi_oversold:
                # Price is below the statistical "cheap" level AND momentum confirms oversold
                position = 1
            elif price > upper and curr_rsi > rsi_overbought:
                # Price is above the statistical "expensive" level AND momentum confirms overbought
                position = -1

        elif position == 1:
            # Long position open — exit once price reverts back above the SMA (fair value)
            if price >= sma:
                position = 0

        elif position == -1:
            # Short position open — exit once price reverts back below the SMA (fair value)
            if price <= sma:
                position = 0

        signals.iloc[i] = position

    return signals


# ─── 5. BACKTESTING ──────────────────────────────────────────────────────────

def backtest(prices: pd.Series, signals: pd.Series) -> pd.DataFrame:
    """
    Run a simple backtest applying yesterday's signal to today's return.

    The one-day shift simulates realistic execution: we generate a signal at
    the close of day t and execute it at the open of day t+1 (approximated
    as the close of day t+1 for this tutorial).

    Args:
        prices:  Series of asset closing prices.
        signals: Series of position signals from generate_signals().

    Returns:
        DataFrame with Price, Signal, daily returns, and cumulative returns.
    """
    daily_returns = prices.pct_change()

    # Shift signals by 1 day: today's signal → tomorrow's return
    strategy_returns = signals.shift(1) * daily_returns

    data = pd.DataFrame(
        {
            "Price": prices,
            "Signal": signals,
            "Market_Return": daily_returns,
            "Strategy_Return": strategy_returns,
        }
    )

    # Cumulative wealth from compounding returns (starting at $1)
    data["Cumulative_Market"] = (1 + data["Market_Return"].fillna(0)).cumprod()
    data["Cumulative_Strategy"] = (1 + data["Strategy_Return"].fillna(0)).cumprod()

    return data


# ─── 6. PERFORMANCE METRICS ──────────────────────────────────────────────────

def performance_summary(results: pd.DataFrame) -> None:
    """
    Print a concise summary of key strategy performance metrics.

    Metrics explained:
    - Total Return:    Compound growth of $1 over the full backtest period.
    - Sharpe Ratio:    Risk-adjusted return = mean_daily_return / std_daily_return * sqrt(252).
                       A Sharpe > 1.0 is generally considered acceptable.
    - Max Drawdown:    Largest peak-to-trough loss in cumulative equity — the "worst case loss".
    - Win Rate:        Fraction of active (non-flat) trading days with a positive return.

    Args:
        results: Output DataFrame from backtest().
    """
    strat_ret = results["Strategy_Return"].fillna(0)
    cum_strat = results["Cumulative_Strategy"]

    total_return = cum_strat.iloc[-1] - 1
    market_return = results["Cumulative_Market"].iloc[-1] - 1

    # Annualised Sharpe Ratio (252 trading days per year)
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else 0.0

    # Max drawdown: track running peak, measure how far current value falls below it
    rolling_peak = cum_strat.cummax()
    drawdown = (cum_strat - rolling_peak) / rolling_peak
    max_drawdown = drawdown.min()

    # Win rate only among days where a position was actually held
    active_days = strat_ret[strat_ret != 0]
    win_rate = (active_days > 0).mean() if len(active_days) > 0 else 0.0

    print("\nPerformance Summary:")
    print(f"  Buy & Hold Return:    {market_return:.2%}")
    print(f"  Strategy Return:      {total_return:.2%}")
    print(f"  Annualised Sharpe:    {sharpe:.2f}")
    print(f"  Max Drawdown:         {max_drawdown:.2%}")
    print(f"  Win Rate (active):    {win_rate:.2%}")
    print(f"  Active Trading Days:  {len(active_days)} / {len(results)}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("MEAN REVERSION STRATEGY – BOLLINGER BANDS + RSI")
    print("=" * 60)

    # Step 1 — Generate synthetic mean-reverting prices via the OU process
    print("\n[1] Generating Ornstein-Uhlenbeck price series (300 days)...")
    prices = generate_mean_reverting_prices(300)
    print(f"  Price range: ${prices.min():.2f} – ${prices.max():.2f}  |  Mean: ${prices.mean():.2f}")

    # Step 2 — Build Bollinger Bands (20-day window, ±2 standard deviations)
    print("[2] Calculating Bollinger Bands (window=20, num_std=2.0)...")
    bands = calculate_bollinger_bands(prices, window=20, num_std=2.0)

    # Step 3 — Compute RSI as a momentum confirmation filter
    print("[3] Calculating RSI (period=14)...")
    rsi = calculate_rsi(prices, period=14)

    # Step 4 — Generate entry/exit signals using dual confirmation
    print("[4] Generating mean-reversion signals (long < lower & RSI<35, short > upper & RSI>65)...")
    signals = generate_signals(prices, bands, rsi)
    long_days = (signals == 1).sum()
    short_days = (signals == -1).sum()
    print(f"  Days long: {long_days}  |  Days short: {short_days}  |  Days flat: {len(signals) - long_days - short_days}")

    # Step 5 — Backtest
    print("[5] Running backtest (signal shifted 1 day to simulate realistic execution)...")
    results = backtest(prices, signals)

    # Step 6 — Show sample output
    print("\nSample Results (last 10 rows):")
    print(results[["Price", "Signal", "Cumulative_Strategy"]].tail(10).to_string())

    # Step 7 — Performance summary
    performance_summary(results)

    print("\nMean Reversion tutorial complete!")
    print("Experiment: Try wider bands (num_std=2.5) or a longer window (30 days).")


if __name__ == "__main__":
    main()
