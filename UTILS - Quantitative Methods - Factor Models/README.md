# Quantitative Methods – Factor Models

## 📋 Overview

Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

## 🎯 Key Concepts

### **Evolution: CAPM → 3-Factor**

| Model | Factors | Year |
|-------|---------|------|
| CAPM | Market (β) | 1964 |
| Fama-French 3F | Market + SMB + HML | 1992 |
| Fama-French 5F | + Profitability (RMW) + Investment (CMA) | 2015 |
| Carhart 4F | + Momentum (UMD) | 1997 |

### **The Three Factors**

**Market (MKT-RF):** Rm − Rf — the return of the overall market above the risk-free rate. This is CAPM's single factor. Every stock has exposure to this.

**SMB (Small Minus Big):** Long small-cap stocks, short large-cap stocks. Historically, smaller companies have delivered higher returns (possibly as compensation for illiquidity and distress risk).

**HML (High Minus Low):** Long value stocks (high Book/Market ratio), short growth stocks. Value stocks have historically outperformed growth (possibly as compensation for financial distress risk or behavioral mispricing).

### **The Factor Model Equation**

```
R_i – RF = α + β_MKT(MKT–RF) + β_SMB·SMB + β_HML·HML + ε
```

| Term | Name | Interpretation |
|------|------|----------------|
| α (alpha) | Intercept | Return unexplained by factors — manager skill or anomaly |
| β_MKT | Market beta | Sensitivity to market-wide moves |
| β_SMB | Size beta | Positive = small-cap tilt; Negative = large-cap tilt |
| β_HML | Value beta | Positive = value tilt; Negative = growth tilt |
| ε | Residual | Idiosyncratic, diversifiable risk |

## 💻 Logic Implemented

1. **Correlated factor simulation** — Multivariate normal with realistic covariance
2. **Stock return generation** — True model + idiosyncratic noise
3. **OLS from scratch** — Matrix algebra: `β = (X'X)⁻¹X'y`
4. **t-statistics** — Statistical significance of each factor loading
5. **Performance attribution** — Decompose average return into factor contributions

## 📂 Files
- `factor_models_tutorial.py`: Factor data generation, OLS regression, significance testing, and performance attribution.

## 🚀 How to Run
```bash
python factor_models_tutorial.py
```

## 🧠 Financial Applications

### 1. Portfolio Risk Decomposition
- "How much of my hedge fund's return is beta to the market vs. true alpha?"
- Investors pay high fees for alpha; beta can be obtained cheaply via ETFs.

### 2. Smart Beta / Factor ETFs
- Deliberately tilting portfolio toward SMB and HML premiums
- Examples: iShares Value ETF (IVE), Dimensional Fund Advisors funds

### 3. Benchmark Construction
- Factor exposures define what a "fair" benchmark for a fund manager is.
- A manager who only buys small-cap value should be benchmarked to small-cap value — not the S&P 500.

### 4. Risk Management
- Stress-test portfolios by shocking factor exposures
- Hedge factor risks using factor ETFs or index futures

### 5. Research — The Zoo of Factors
- Academic literature has identified 300+ potential factors.
- Most don't survive out-of-sample. The three Fama-French factors are among the most robust.

## 💡 Best Practices

- **Use monthly data**: Daily factor returns are noisier; monthly gives better signal-to-noise for factor regressions.
- **Check R-squared**: < 0.40 suggests the factors don't explain this stock well (could be a niche sector or anomaly).
- **Multiple testing**: With 300+ factors available, any single significant result could be spurious — use Bonferroni correction or out-of-sample tests.
- **Factor stability**: Betas change over time as companies grow or shift strategy — use rolling regressions to monitor.
- **Real data**: Download free Fama-French factor returns from [Ken French's website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
