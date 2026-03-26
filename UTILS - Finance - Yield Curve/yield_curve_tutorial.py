"""Yield Curve Construction and Analysis.

Run with:
    python yield_curve_tutorial.py

The yield curve is perhaps the single most important chart in all of finance.
It shows the relationship between interest rates (yields) and time to maturity
for bonds of equal credit quality (e.g., US Treasuries).

This tutorial covers three approaches:
  1. Interpreting raw par yields from the market (the most common starting point).
  2. Fitting the Nelson-Siegel (1987) parametric model to smooth the curve.
  3. Extracting implied forward rates from the smoothed spot (zero) curve.

Key Concepts:
- Par Yield:    The coupon rate making a bond price equal to face value ($100).
- Zero/Spot Rate: The yield on a zero-coupon bond — the "pure" rate for that maturity.
- Forward Rate:  The implied rate for a *future* period, derived from today's spot curve.
- Inverted Curve: Short rates > long rates — historically the most reliable recession indicator.
- Nelson-Siegel: A 4-parameter model capturing level, slope, and curvature of the yield curve.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ─── 1. RAW YIELD DATA ──────────────────────────────────────────────────────


def get_sample_treasury_yields() -> pd.DataFrame:
    """
    Return a representative set of US Treasury par yields.

    In practice you would fetch these from FRED (Federal Reserve Economic Data),
    Bloomberg, or a broker API.  Here we use approximate values from a "normal"
    (upward-sloping) yield curve so the tutorial is fully self-contained.

    The data reflects a high-rate environment (similar to 2023–2024 US Treasuries)
    where short-end rates are elevated and the curve is mildly inverted at the front.

    Returns:
        DataFrame with columns:
            maturity    – time to maturity in years (float)
            yield_pct   – par yield in percentage points (e.g., 5.3 means 5.3%)
    """
    data = {
        # Maturity labels: 1m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y
        "maturity": [1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30],
        "yield_pct": [5.30, 5.35, 5.40, 5.20, 4.80, 4.55, 4.30, 4.25, 4.20, 4.40, 4.30],
    }
    return pd.DataFrame(data)


# ─── 2. NELSON-SIEGEL MODEL ──────────────────────────────────────────────────


def nelson_siegel(maturity: np.ndarray, beta0: float, beta1: float, beta2: float, tau: float) -> np.ndarray:
    """
    Compute Nelson-Siegel fitted yields for the given maturities.

    The Nelson-Siegel (1987) model decomposes the yield curve into three additive factors:

        y(T) = beta0
             + beta1 * [(1 – exp(–T/tau)) / (T/tau)]
             + beta2 * [(1 – exp(–T/tau)) / (T/tau) – exp(–T/tau)]

    Economic interpretation of each parameter:
    - beta0 (Level):     Long-run yield; the curve asymptote as T → ∞.
                          Increase beta0 → shift the whole curve up or down.
    - beta1 (Slope):     Short-to-long rate difference.
                          Negative beta1 → normal (upward-sloping) curve.
                          Positive beta1 → inverted curve (short > long).
    - beta2 (Curvature): Hump (or trough) at intermediate maturities.
                          Positive beta2 → hump; negative beta2 → U-shape.
    - tau   (Decay):     Controls where the slope and curvature loadings peak.
                          Larger tau → hump occurs at longer maturities.

    Args:
        maturity: Array of maturities in years (strictly positive).
        beta0:    Level factor (long-run rate, in the same units as the output).
        beta1:    Slope factor.
        beta2:    Curvature factor.
        tau:      Decay parameter in years (controls hump location).

    Returns:
        Array of fitted yields in the same units as beta0/beta1/beta2.
    """
    # Guard against division by zero for very short maturities
    T = np.maximum(maturity, 1e-6)
    t_tau = T / tau

    # The loading function (1 - e^(-T/tau)) / (T/tau) is shared by slope and curvature
    loading = (1 - np.exp(-t_tau)) / t_tau

    y = beta0 + beta1 * loading + beta2 * (loading - np.exp(-t_tau))
    return y


def fit_nelson_siegel(maturities: np.ndarray, yields_pct: np.ndarray) -> dict:
    """
    Fit Nelson-Siegel parameters to observed par yields via numerical least-squares.

    We minimise the sum of squared differences between observed (market) yields
    and the Nelson-Siegel model yields.  scipy.optimize.minimize with the
    Nelder-Mead method works well here because:
    - The objective is smooth but the parameter space has a ridge (tau can vary widely).
    - Nelder-Mead is derivative-free and robust for small 4-parameter problems.

    Args:
        maturities: Array of bond maturities in years (must be strictly positive).
        yields_pct: Observed par yields in percent (e.g., 4.5 for 4.5%).

    Returns:
        Dict with keys: beta0, beta1, beta2, tau (fitted parameters), and
        rmse (root-mean-square fitting error in basis points equivalent).
    """

    def objective(params: np.ndarray) -> float:
        """Sum of squared deviations between model and market yields."""
        b0, b1, b2, t = params
        if t <= 0:
            # tau must be positive — return a large penalty for invalid values
            return 1e9
        fitted = nelson_siegel(maturities, b0, b1, b2, t)
        return float(np.sum((fitted - yields_pct) ** 2))

    # Initial guess calibrated for a ~5% US Treasury environment
    x0 = [4.5, -1.0, 1.5, 2.0]  # [beta0, beta1, beta2, tau]

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
    )

    b0, b1, b2, tau = result.x
    fitted_yields = nelson_siegel(maturities, b0, b1, b2, tau)
    rmse = float(np.sqrt(np.mean((fitted_yields - yields_pct) ** 2)))

    return {"beta0": b0, "beta1": b1, "beta2": b2, "tau": tau, "rmse": rmse}


# ─── 3. FORWARD RATE EXTRACTION ──────────────────────────────────────────────


def compute_forward_rates(maturities: np.ndarray, spot_rates_pct: np.ndarray) -> pd.DataFrame:
    """
    Compute implied instantaneous forward rates from a set of spot (zero) rates.

    The forward rate f(T1, T2) is the rate agreed today for borrowing between
    time T1 and T2 in the future.  It is implied by the no-arbitrage condition:

        investing $1 from 0→T2 at spot rate r(T2) must equal
        investing $1 from 0→T1 at r(T1) and then from T1→T2 at f(T1,T2).

    Under continuous compounding:
        f(T1, T2) = [r(T2) * T2 – r(T1) * T1] / (T2 – T1)

    Practical interpretation:
    - If the forward rate for years 9–10 is 3.5%, the market implies that
      short-term rates in 9 years will be approximately 3.5%.
    - An upward-sloping forward curve → market expects rates to rise.
    - A downward-sloping forward curve → market expects rates to fall (or recession).

    Args:
        maturities:     Array of maturities in years (strictly increasing).
        spot_rates_pct: Corresponding spot (zero) rates in percent.

    Returns:
        DataFrame with columns: Maturity_Yrs, Spot_Rate_Pct, Forward_Rate_Pct.
    """
    # Convert percent to decimal for the arithmetic
    r = spot_rates_pct / 100
    T = maturities

    forward_rates = np.zeros(len(T))
    # The "forward rate" for the first bucket equals the spot rate (no prior period)
    forward_rates[0] = r[0]

    for i in range(1, len(T)):
        # Implied rate for the incremental period (T[i-1] → T[i])
        dt = T[i] - T[i - 1]
        forward_rates[i] = (r[i] * T[i] - r[i - 1] * T[i - 1]) / dt

    return pd.DataFrame(
        {
            "Maturity_Yrs": T,
            "Spot_Rate_Pct": spot_rates_pct,
            "Forward_Rate_Pct": forward_rates * 100,  # back to percent
        }
    )


# ─── 4. CURVE SHAPE CLASSIFICATION ──────────────────────────────────────────


def classify_curve_shape(short_rate: float, long_rate: float, mid_rate: float) -> str:
    """
    Classify the shape of the yield curve and its macro-economic implication.

    The yield curve shape is the most widely watched macroeconomic indicator
    because it embeds the market's collective expectation of future growth and
    interest rate policy.

    Shape classification based on the 2-year / 10-year spread:
    - Normal   (spread > +30 bps): Long rates above short rates → growth expected.
    - Inverted (spread < –10 bps): Short rates above long rates → recession signal.
      (The 2s10s inversion has preceded every US recession since 1955.)
    - Flat     (spread near zero): Transition between normal and inverted.
    - Humped   (mid > both ends): Complex rate expectations; often mid-cycle.

    Args:
        short_rate: Yield at the short end (e.g., 2-year Treasury yield).
        long_rate:  Yield at the long end (e.g., 10-year Treasury yield).
        mid_rate:   Yield at an intermediate point (e.g., 5-year).

    Returns:
        String describing the curve shape and its economic implication.
    """
    spread = long_rate - short_rate  # the "2s10s spread" in bond market parlance

    if spread > 0.30:
        return "Normal (Upward-Sloping) — Economy expected to grow"
    elif spread < -0.10:
        return "Inverted (Downward-Sloping) — Recession warning signal"
    elif abs(spread) <= 0.30 and mid_rate > long_rate and mid_rate > short_rate:
        return "Humped — Peak at intermediate maturities"
    else:
        return "Flat — Transition period; uncertainty about future rates"


# ─── MAIN ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("YIELD CURVE CONSTRUCTION & ANALYSIS")
    print("=" * 60)

    # ── Step 1: Display raw market yield data ──
    print("\n[1] Sample US Treasury Par Yields (normal/slight inversion environment):")
    df = get_sample_treasury_yields()

    for _, row in df.iterrows():
        # Format maturity as months for sub-1-year, years otherwise
        if row["maturity"] < 1:
            mat_label = f"{row['maturity'] * 12:.0f}mo"
        else:
            mat_label = f"{row['maturity']:.0f}yr"
        print(f"  {mat_label:>5}  →  {row['yield_pct']:.2f}%")

    maturities = df["maturity"].values
    yields = df["yield_pct"].values

    # ── Step 2: Classify the curve shape ──
    print("\n[2] Yield Curve Shape Analysis:")
    # Use 2yr, 5yr, 10yr as the short/mid/long reference points
    y_2y = float(df.loc[df["maturity"] == 2, "yield_pct"].iloc[0])
    y_5y = float(df.loc[df["maturity"] == 5, "yield_pct"].iloc[0])
    y_10y = float(df.loc[df["maturity"] == 10, "yield_pct"].iloc[0])

    spread_2s10s = y_10y - y_2y
    shape = classify_curve_shape(y_2y, y_10y, y_5y)
    print(f"  2yr: {y_2y:.2f}%  |  5yr: {y_5y:.2f}%  |  10yr: {y_10y:.2f}%")
    print(f"  2s10s Spread: {spread_2s10s:+.2f}%  ({spread_2s10s * 100:+.0f} bps)")
    print(f"  Shape: {shape}")

    # ── Step 3: Fit Nelson-Siegel model ──
    print("\n[3] Fitting Nelson-Siegel Model to smooth the curve...")
    params = fit_nelson_siegel(maturities, yields)
    print(f"  beta0 (Level):      {params['beta0']:.4f}%   ← long-run yield")
    print(f"  beta1 (Slope):      {params['beta1']:.4f}%   ← short vs long rate difference")
    print(f"  beta2 (Curvature):  {params['beta2']:.4f}%   ← hump magnitude")
    print(f"  tau   (Decay):      {params['tau']:.4f} yr  ← hump location")
    print(f"  RMSE  (fit error):  {params['rmse']:.4f}%   ← smaller = better fit")

    # ── Step 4: Extract forward rates from the smoothed Nelson-Siegel curve ──
    print("\n[4] Implied Forward Rates (from Nelson-Siegel fitted curve):")
    smooth_mats = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    smooth_yields = nelson_siegel(smooth_mats, params["beta0"], params["beta1"], params["beta2"], params["tau"])
    fwd_df = compute_forward_rates(smooth_mats, smooth_yields)
    print(fwd_df.to_string(index=False, float_format="{:.3f}".format))

    # ── Step 5: Model vs. market comparison ──
    print("\n[5] Nelson-Siegel Model vs. Market Yields (fit quality):")
    print(f"  {'Maturity':>10}  {'Market':>10}  {'Model':>10}  {'Error (bps)':>12}")
    print("  " + "-" * 46)
    for _, row in df.iterrows():
        model_y = nelson_siegel(
            np.array([row["maturity"]]), params["beta0"], params["beta1"], params["beta2"], params["tau"]
        )[0]
        error_bps = (model_y - row["yield_pct"]) * 100
        mat_str = f"{row['maturity']:.2f}yr"
        print(f"  {mat_str:>10}  {row['yield_pct']:>10.3f}%  {model_y:>10.3f}%  {error_bps:>+11.1f}")

    print("\nYield Curve tutorial complete!")
    print("Tip: Try constructing an inverted curve by swapping short/long rates.")
    print("     The 2s10s spread turning negative has preceded every US recession since 1955.")


if __name__ == "__main__":
    main()
