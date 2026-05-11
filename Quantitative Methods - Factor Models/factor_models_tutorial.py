"""Multi-Factor Models: Fama-French 3-Factor Tutorial.

Run with:
    python factor_models_tutorial.py

The Capital Asset Pricing Model (CAPM, 1964) was a landmark: it explained stock
returns using a single factor (the overall market return).  But CAPM left a lot
of return unexplained.  Eugene Fama and Kenneth French (1992) proposed a 3-factor
model adding Size (SMB) and Value (HML) as additional systematic risk factors.

Understanding factor models is essential for:
- Portfolio risk decomposition (how much return comes from factors vs. manager skill?)
- Performance attribution (separating alpha from beta exposures)
- Factor / smart-beta investing (deliberately tilting toward rewarded risk premia)
- Risk management (hedging specific factor exposures)

Key Concepts:
- Alpha (α): Return not explained by any factor — pure manager skill or unexploited anomaly.
- Beta (β):  Sensitivity of a stock's excess return to a given factor.
- SMB:       Small Minus Big — long small-cap, short large-cap. Captures the size premium.
- HML:       High Minus Low — long value (high book-to-market), short growth stocks.
- R-Squared: Fraction of return variance explained by the model; higher = better coverage.
- t-Statistic: |t| > 2 ≈ coefficient is statistically significant at the 95% confidence level.
"""

import numpy as np
import pandas as pd

# ─── 1. SYNTHETIC FACTOR DATA ────────────────────────────────────────────────


def generate_synthetic_factor_data(n_periods: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic monthly factor return data resembling Fama-French factors.

    In production you would download real factor data for free from Kenneth French's
    library at Dartmouth: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

    Here we simulate data with realistic statistical properties using a multivariate
    normal distribution with an estimated correlation structure.

    Historical approximate monthly statistics:
    - MKT-RF: mean ≈ +0.50%/mo, std ≈ 4.5%/mo  (market excess return)
    - SMB:    mean ≈ +0.20%/mo, std ≈ 2.5%/mo  (small-cap premium)
    - HML:    mean ≈ +0.30%/mo, std ≈ 2.5%/mo  (value premium)
    - RF:     mean ≈ +0.03%/mo, std ≈ 0.01%/mo (Treasury bill rate)

    Args:
        n_periods: Number of monthly observations (120 = 10 years of data).
        seed:      Random seed for reproducibility.

    Returns:
        DataFrame with columns: MKT_RF, SMB, HML, RF (all in % per month).
    """
    rng = np.random.default_rng(seed)

    # Correlation structure between the three factors
    # Market and HML are mildly negatively correlated (growth stocks drive market cap-weighted index)
    # SMB and HML have low correlation
    corr_matrix = np.array(
        [
            [1.00, 0.20, -0.30],  # MKT_RF row: correlations with SMB, HML
            [0.20, 1.00, 0.05],  # SMB row
            [-0.30, 0.05, 1.00],  # HML row
        ]
    )

    # Monthly standard deviations in percent
    stds = np.array([4.5, 2.5, 2.5])
    # Monthly mean returns in percent
    means = np.array([0.5, 0.2, 0.3])

    # Covariance matrix = diag(std) × corr × diag(std)
    cov_matrix = np.diag(stds) @ corr_matrix @ np.diag(stds)

    # Draw correlated factor returns from multivariate normal
    factors = rng.multivariate_normal(means, cov_matrix, size=n_periods)

    # Risk-free rate: near-constant with minimal variation (short-term T-bill)
    rf = rng.normal(0.03, 0.01, size=n_periods)

    return pd.DataFrame(
        {
            "MKT_RF": factors[:, 0],  # Market excess return: Rm - Rf
            "SMB": factors[:, 1],  # Small Minus Big: R_small - R_big
            "HML": factors[:, 2],  # High Minus Low: R_value - R_growth
            "RF": rf,  # Risk-free rate (e.g., 3-month T-bill)
        }
    )


# ─── 2. SYNTHETIC STOCK RETURNS ──────────────────────────────────────────────


def generate_stock_returns(
    factor_data: pd.DataFrame,
    alpha: float,
    beta_mkt: float,
    beta_smb: float,
    beta_hml: float,
    idio_vol: float,
    seed: int = 0,
) -> pd.Series:
    """
    Generate stock excess returns implied by the Fama-French 3-Factor model.

    The model equation is:
        R_i – RF = alpha + beta_MKT*(MKT–RF) + beta_SMB*SMB + beta_HML*HML + epsilon

    Interpretation of each term:
    - alpha:     Return per period not explained by any factor.
                 In efficient markets this should be close to zero.
    - beta_MKT:  How much the stock amplifies or dampens overall market moves.
                 > 1 = aggressive, < 1 = defensive, < 0 = inverse (rare).
    - beta_SMB:  Positive → tilted toward small-cap stocks (higher SMB factor loading).
                 Negative → tilted toward large-cap stocks.
    - beta_HML:  Positive → tilted toward value stocks (cheap on book value).
                 Negative → tilted toward growth stocks (expensive on book value).
    - epsilon:   Idiosyncratic noise specific to this stock, uncorrelated with any factor.
                 This is the risk that diversification *can* eliminate.

    Args:
        factor_data: DataFrame from generate_synthetic_factor_data().
        alpha:       Monthly alpha contribution in %.
        beta_mkt:    Market (systematic) beta.
        beta_smb:    Size factor loading.
        beta_hml:    Value factor loading.
        idio_vol:    Monthly idiosyncratic (stock-specific) volatility in %.
        seed:        Random seed for the noise term.

    Returns:
        pd.Series of monthly stock excess returns (stock return minus risk-free rate) in %.
    """
    rng = np.random.default_rng(seed)

    # Systematic return: the part of the stock's return explained by the three factors
    factor_return = beta_mkt * factor_data["MKT_RF"] + beta_smb * factor_data["SMB"] + beta_hml * factor_data["HML"]

    # Idiosyncratic return: noise specific to this company (earnings surprises, news, etc.)
    # This component is uncorrelated with any factor and can be diversified away
    epsilon = rng.normal(0.0, idio_vol, size=len(factor_data))

    excess_return = alpha + factor_return + epsilon
    return pd.Series(excess_return, name="Excess_Return")


# ─── 3. OLS REGRESSION ───────────────────────────────────────────────────────


def run_ols_regression(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    Run Ordinary Least Squares (OLS) regression from scratch using linear algebra.

    OLS finds the coefficient vector beta that minimises the sum of squared residuals:
        min_beta  ||y – X*beta||^2

    The closed-form (analytical) solution is:
        beta_hat = (X'X)^{-1} X'y

    This is exactly what statsmodels, sklearn.linear_model.LinearRegression,
    and Excel's LINEST() function compute under the hood.

    Additional statistics:
    - R-squared:    1 – RSS/TSS; fraction of y variance explained by X.
    - Residual std: sqrt(RSS / (n–k)); standard deviation of unexplained returns.
    - t-statistic:  beta_i / se(beta_i); how many standard errors beta is from zero.
                    |t| > 1.96 → significant at 95%;  |t| > 2.58 → significant at 99%.

    Args:
        y: Dependent variable (stock excess returns as a pd.Series).
        X: Design matrix including a constant column for the intercept (alpha).
           Columns should be: ['const', 'MKT_RF', 'SMB', 'HML'].

    Returns:
        Dict with: coefficients, t_stats, r_squared, residual_std, n_obs.
    """
    y_arr = y.values.reshape(-1, 1)
    X_arr = X.values

    # ── OLS closed form: beta = (X'X)^{-1} X'y ──
    XtX = X_arr.T @ X_arr
    Xty = X_arr.T @ y_arr

    # lstsq is more numerically stable than explicit matrix inverse
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0].flatten()

    # ── Residuals and variance ──
    y_hat = X_arr @ beta
    residuals = y_arr.flatten() - y_hat

    n = len(y_arr)  # number of observations
    k = X_arr.shape[1]  # number of parameters (including intercept)

    # Residual sum of squares / degrees of freedom = unbiased variance estimate
    rss = float(np.sum(residuals**2))
    sigma2 = rss / (n - k)

    # Total sum of squares for R-squared
    tss = float(np.sum((y_arr.flatten() - y_arr.mean()) ** 2))
    r_squared = 1.0 - rss / tss

    # ── Standard errors and t-statistics ──
    # Var(beta_hat) = sigma^2 * (X'X)^{-1}
    var_beta = sigma2 * np.linalg.inv(XtX)
    se_beta = np.sqrt(np.diag(var_beta))

    # t-statistic: test whether each coefficient is significantly different from zero
    t_stats = beta / se_beta

    return {
        "coefficients": dict(zip(X.columns, beta)),
        "t_stats": dict(zip(X.columns, t_stats)),
        "r_squared": r_squared,
        "residual_std": float(np.sqrt(sigma2)),
        "n_obs": n,
    }


# ─── 4. RESULTS DISPLAY ──────────────────────────────────────────────────────


def print_regression_table(reg_result: dict, label: str) -> None:
    """
    Print a formatted regression results table.

    Significance stars follow the standard econometric convention:
    - ***  p < 0.01  (|t| > 2.58)  — highly significant
    - *    p < 0.05  (|t| > 1.96)  — significant
    - (blank)        p > 0.05      — not significant at conventional levels

    Args:
        reg_result: Dict output from run_ols_regression().
        label:      Description of the stock/portfolio being analysed.
    """
    coefs = reg_result["coefficients"]
    tstats = reg_result["t_stats"]

    print(f"\n  {'─' * 52}")
    print(f"  Regression: {label}")
    print(f"  {'─' * 52}")
    print(f"  R-Squared:    {reg_result['r_squared']:.4f}  (fraction of variance explained)")
    print(f"  Residual Std: {reg_result['residual_std']:.4f}%/month (idiosyncratic risk)")
    print(f"  Observations: {reg_result['n_obs']}")
    print()

    # Map internal column names to human-readable labels
    factor_names = {
        "const": "Alpha (α)",
        "MKT_RF": "Market Beta",
        "SMB": "SMB Beta (Size)",
        "HML": "HML Beta (Value)",
    }

    header = f"  {'Factor':<18}  {'Coeff':>10}  {'t-stat':>8}  {'Sig':>5}"
    print(header)
    print("  " + "-" * 46)

    for col, coef in coefs.items():
        t = tstats[col]
        sig = "***" if abs(t) > 2.58 else ("*" if abs(t) > 1.96 else "")
        name = factor_names.get(col, col)
        print(f"  {name:<18}  {coef:>10.4f}  {t:>8.2f}  {sig:>5}")


# ─── 5. FACTOR CONTRIBUTION ANALYSIS ────────────────────────────────────────


def factor_contribution(reg_result: dict, factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the average monthly excess return into factor contributions.

    This shows how much of a stock's historical return came from each factor
    exposure versus unexplained alpha.  The decomposition follows:

        E[R_excess] = alpha + beta_MKT * E[MKT_RF] + beta_SMB * E[SMB] + beta_HML * E[HML]

    This is the foundation of *performance attribution* — understanding *why*
    a portfolio generated its returns (factor exposures vs. manager skill).

    Args:
        reg_result:  Dict output from run_ols_regression().
        factor_data: DataFrame with MKT_RF, SMB, HML columns.

    Returns:
        DataFrame with factor, beta, factor_mean, and contribution columns.
    """
    coefs = reg_result["coefficients"]

    factor_means = {
        "const": 1.0,  # constant's "mean" is 1 by definition
        "MKT_RF": factor_data["MKT_RF"].mean(),
        "SMB": factor_data["SMB"].mean(),
        "HML": factor_data["HML"].mean(),
    }

    rows = []
    for col, beta in coefs.items():
        mean_val = factor_means[col]
        contribution = beta * mean_val
        factor_names = {
            "const": "Alpha",
            "MKT_RF": "Market (MKT-RF)",
            "SMB": "Size (SMB)",
            "HML": "Value (HML)",
        }
        rows.append(
            {
                "Factor": factor_names.get(col, col),
                "Beta": beta,
                "Factor_Mean_%": mean_val if col != "const" else float("nan"),
                "Contribution_%": contribution,
            }
        )

    return pd.DataFrame(rows)


# ─── MAIN ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("FAMA-FRENCH 3-FACTOR MODEL TUTORIAL")
    print("=" * 60)

    # ── Step 1: Generate 10 years of monthly factor data ──
    print("\n[1] Generating 10 years (120 months) of synthetic factor data...")
    factors = generate_synthetic_factor_data(n_periods=120)

    print(
        f"  Factor means (per month):  "
        f"MKT={factors['MKT_RF'].mean():.2f}%  "
        f"SMB={factors['SMB'].mean():.2f}%  "
        f"HML={factors['HML'].mean():.2f}%"
    )
    print(
        f"  Factor std   (per month):  "
        f"MKT={factors['MKT_RF'].std():.2f}%  "
        f"SMB={factors['SMB'].std():.2f}%  "
        f"HML={factors['HML'].std():.2f}%"
    )

    # ── Step 2: Simulate two stocks with different factor profiles ──
    print("\n[2] Simulating two stocks with distinct factor exposures:")

    # Stock A: "Growth" stock — high market beta, negative HML (not cheap on book value)
    print("  Stock A: High-beta growth stock (large-cap tech-like)")
    print("    → alpha=0.05%, beta_mkt=1.2, beta_smb=–0.3 (large-cap), beta_hml=–0.5 (growth)")
    ret_a = generate_stock_returns(
        factors, alpha=0.05, beta_mkt=1.2, beta_smb=-0.3, beta_hml=-0.5, idio_vol=3.0, seed=1
    )

    # Stock B: "Value small-cap" — positive SMB (small) and HML (cheap book value), lower market beta
    print("  Stock B: Small-cap value stock (cheap, under-followed)")
    print("    → alpha=0.10%, beta_mkt=0.8, beta_smb=+0.6 (small-cap), beta_hml=+0.7 (value)")
    ret_b = generate_stock_returns(factors, alpha=0.10, beta_mkt=0.8, beta_smb=0.6, beta_hml=0.7, idio_vol=3.5, seed=2)

    # ── Step 3: Prepare the design matrix ──
    # The design matrix X has 4 columns: constant (for alpha), MKT_RF, SMB, HML
    X = factors[["MKT_RF", "SMB", "HML"]].copy()
    X.insert(0, "const", 1.0)  # intercept column → alpha in factor model language

    # ── Step 4: Run OLS factor regressions ──
    print("\n[3] Running Fama-French 3-Factor Regressions:")
    result_a = run_ols_regression(ret_a, X)
    print_regression_table(result_a, "Stock A – Growth (High-beta, Large-cap)")

    result_b = run_ols_regression(ret_b, X)
    print_regression_table(result_b, "Stock B – Value Small-Cap")

    # ── Step 5: Performance attribution ──
    print("\n[4] Performance Attribution (avg monthly return decomposition):")

    for label, result in [("Stock A", result_a), ("Stock B", result_b)]:
        contrib_df = factor_contribution(result, factors)
        total = contrib_df["Contribution_%"].sum()
        print(f"\n  {label}  (total explained avg return: {total:.4f}%/month)")
        print(contrib_df.to_string(index=False, float_format="{:.4f}".format))

    # ── Step 6: Economic interpretation ──
    print("\n[5] Economic Interpretation:")
    print("  Stock A negative HML beta → growth stock (market prices future earnings growth,")
    print("    not current book value).")
    print("  Stock B positive SMB & HML betas → exploits the Size and Value premiums.")
    print("  High R-squared → most return variation is captured by just 3 factors.")
    print("  Alpha close to true values (0.05%, 0.10%) given 120 monthly observations.")
    print()
    print("  Factor investing: if SMB and HML are persistent risk premia,")
    print("  simply tilting a portfolio toward small-cap and value stocks captures them")
    print("  without requiring stock-picking skill (alpha).")

    print("\nFactor Models tutorial complete!")
    print("Tip: Download real Fama-French data from Ken French's Dartmouth library (free).")


if __name__ == "__main__":
    main()
