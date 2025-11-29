# Quantitative Methods – Optimization

## 📋 Overview

Optimization is the mathematical engine behind modern finance. From finding the best portfolio weights to calibrating complex models, optimization techniques are essential for quantitative analysts.

## 🎯 Key Concepts

### **Minimization**
- **Objective Function**: The quantity to minimize (e.g., risk, error)
- **Constraints**: Limitations (e.g., weights sum to 1, long-only)
- **Bounds**: Limits on individual variables (e.g., 0% to 100% allocation)
- **Algorithms**: BFGS, SLSQP, Newton-CG

### **Portfolio Optimization**
- **Mean-Variance Optimization**: Markowitz Modern Portfolio Theory
- **Global Minimum Variance**: Portfolio with lowest possible risk
- **Tangency Portfolio**: Portfolio with highest Sharpe Ratio
- **Efficient Frontier**: Set of optimal portfolios

### **Curve Fitting**
- **Calibration**: Fitting model parameters to market data
- **Least Squares**: Minimizing the sum of squared errors
- **Yield Curve Modeling**: Nelson-Siegel, Svensson models

### **Root Finding**
- **Solving Equations**: Finding x where f(x) = 0
- **Implied Volatility**: Finding volatility that matches option price
- **IRR**: Finding discount rate where NPV = 0

## 💻 Key Examples

### Basic Minimization
```python
from scipy.optimize import minimize

def objective(x):
    return (x[0] - 3)**2 + 5

result = minimize(objective, [0])
print(result.x)  # [3.0]
```

### Portfolio Optimization
```python
# Minimize volatility subject to constraints
def volatility(weights):
    return np.sqrt(weights.T @ cov_matrix @ weights)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = ((0, 1), (0, 1), (0, 1))

result = minimize(volatility, init_guess, 
                 method='SLSQP', bounds=bounds, constraints=constraints)
```

### Implied Volatility (Root Finding)
```python
from scipy.optimize import newton

def objective(sigma):
    return bs_call(S, K, T, r, sigma) - market_price

iv = newton(objective, x0=0.2)
```

## 📂 Files
- `optimization_tutorial.py`: Comprehensive optimization tutorial

## 🚀 How to Run
```bash
pip install scipy numpy
python optimization_tutorial.py
```

## 🧠 Financial Applications

### 1. Portfolio Construction
- **Risk Parity**: Equal risk contribution from each asset
- **Black-Litterman**: Combining market equilibrium with views
- **Index Tracking**: Minimizing tracking error vs benchmark

### 2. Model Calibration
- **Yield Curves**: Fitting Nelson-Siegel to bond prices
- **Volatility Surfaces**: Fitting SVI model to option chains
- **Local Volatility**: Calibrating Dupire's formula

### 3. Algorithmic Trading
- **Parameter Tuning**: Optimizing strategy parameters (stop loss, window size)
- **Execution Algorithms**: Minimizing market impact (VWAP/TWAP)
- **Arbitrage**: Finding optimal basket for stat arb

## 💡 Best Practices

- **Convexity**: Convex problems (like variance minimization) have unique global solutions. Non-convex problems may get stuck in local minima.
- **Scaling**: Ensure variables are on similar scales for better convergence.
- **Constraints**: Use 'SLSQP' or 'trust-constr' for constrained problems.
- **Initial Guess**: A good starting point improves speed and reliability.

## 📚 References

- **SciPy Optimize**: https://docs.scipy.org/doc/scipy/reference/optimize.html
- **Convex Optimization**: Boyd & Vandenberghe
- **Portfolio Optimization**: Cornuejols & Tutuncu

---

*Master optimization to solve the most challenging problems in quantitative finance!*
