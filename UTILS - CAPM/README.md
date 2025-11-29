# CAPM (Capital Asset Pricing Model) Utility

This module lets you calculate the expected return of any stock or portfolio according to CAPM, a core idea in modern finance for pricing risky assets.

## What is CAPM?
- CAPM gives you the *expected* return of an asset based on its risk relative to the market (called *beta*), the risk-free rate, and the market’s return.
- It’s used by professionals and students to estimate if an investment is fairly compensated for its risk.

**Formula:**
> Expected Return = Risk-Free Rate + Beta × (Market Return - Risk-Free Rate)

## How to Use
1. Put your numbers into the `capm_expected_return()` function in `capm_calculator.py`.
    - risk_free_rate = annual risk-free rate, e.g., yield on a government bond (as decimal, 3% = 0.03)
    - beta = sensitivity of asset to market, usually found on finance websites
    - market_return = expected annual market return (as decimal)
2. Run the script or import the function in your project.

### Quick Example
```python
from capm_calculator import capm_expected_return
rf = 0.03
beta = 1.2
rm = 0.09
print('CAPM Return:', capm_expected_return(rf, beta, rm))
```

## Why It’s Useful
- Helps judge if an asset is worth its risk
- Core to portfolio theory and investing basics
- Used in professional valuation, risk management, and exam syllabi
