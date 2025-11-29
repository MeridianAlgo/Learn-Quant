# Bond Price and Yield Calculator

This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

## What is a Bond?
- A bond is a type of loan you give to a company or government. In return, they pay you interest ("coupons") regularly, and repay the face value at maturity.
- Bonds are a huge part of financial markets, used by everyone from governments to big investors.

## Key Formulas
- **Bond Price:** Present value of all future coupon and face value payments, discounted at the yield to maturity (YTM).
- **Yield to Maturity (YTM):** The effective annual return you'd earn if you buy the bond today and hold to maturity.

## How to Use
1. Use `bond_price()` to find fair value given face, coupon, periods, and YTM.
2. Use `estimate_ytm()` to estimate the yield given price, face, coupon, and periods.

### Example
```python
from bond_tools import bond_price, estimate_ytm
print('Bond value:', bond_price(1000, 0.05, 10, 0.04))
print('Estimated YTM:', estimate_ytm(1000, 0.05, 10, 1050))
```

## Why It Matters
- Bonds are a safe and steady part of many portfolios
- Bankers, exam takers, and investors all need these calculations
- Helps you understand time value of money and how interest rates affect prices!

*Explore more in UTILS and Documentation folders!*
