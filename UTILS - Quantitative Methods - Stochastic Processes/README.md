# Quantitative Methods – Stochastic Processes

## 📋 Overview

Stochastic processes are mathematical models for random systems evolving over time. In finance, they are used to model asset prices, interest rates, and volatility for pricing derivatives and managing risk.

## 🎯 Key Concepts

### **Brownian Motion (Wiener Process)**
- **Random Walk**: Continuous-time version of a random walk
- **Properties**:
  - Starts at 0
  - Independent increments
  - Gaussian increments: $W_{t+u} - W_t \sim N(0, u)$
  - Continuous paths but nowhere differentiable (fractal)

### **Geometric Brownian Motion (GBM)**
- **Stock Prices**: Standard model for equities (Black-Scholes)
- **Log-Normal**: Prices cannot be negative
- **Equation**: $dS_t = \mu S_t dt + \sigma S_t dW_t$
- **Solution**: $S_t = S_0 \exp((\mu - 0.5\sigma^2)t + \sigma W_t)$

### **Mean Reversion (Ornstein-Uhlenbeck)**
- **Pull to Mean**: Prices tend to return to a long-term average
- **Use Cases**: Volatility, interest rates, spread trading
- **Equation**: $dx_t = \theta(\mu - x_t)dt + \sigma dW_t$
- **$\theta$**: Speed of mean reversion

### **Jump Diffusion**
- **Fat Tails**: Models sudden market shocks (crashes, news)
- **Components**: GBM (continuous) + Poisson Jumps (discontinuous)
- **Merton Model**: Jumps are log-normally distributed

## 💻 Key Examples

### Simulating GBM
```python
# S(t) = S(0) * exp((μ - 0.5σ²)t + σW(t))
drift = (mu - 0.5 * sigma**2) * dt
diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1, N)
price_path = S0 * np.exp(np.cumsum(drift + diffusion))
```

### Simulating Mean Reversion
```python
# Euler-Maruyama discretization
for t in range(1, N):
    dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    x[t] = x[t-1] + dx
```

## 📂 Files
- `stochastic_tutorial.py`: Interactive tutorial with simulations

## 🚀 How to Run
```bash
pip install numpy matplotlib
python stochastic_tutorial.py
```

## 🧠 Financial Applications

### 1. Option Pricing (Monte Carlo)
Simulate thousands of price paths using GBM to price complex derivatives (e.g., Asian options, Barrier options) where analytical formulas don't exist.

### 2. Risk Management (VaR)
Use simulations to estimate Value at Risk (VaR) and Expected Shortfall (CVaR) by generating potential future portfolio values.

### 3. Pairs Trading
Model the spread between two correlated assets as an Ornstein-Uhlenbeck process. Trade when the spread deviates significantly from the mean (buy low, sell high).

### 4. Volatility Modeling
Stochastic volatility models (like Heston) assume volatility itself follows a stochastic process (often mean-reverting).

## 💡 Best Practices

- **Seed Random Numbers**: Always use `np.random.seed()` for reproducible results.
- **Time Steps**: Use sufficiently small `dt` for accuracy, especially for mean-reverting processes.
- **Vectorization**: Use NumPy vectorization instead of loops for GBM simulations to speed up calculation by 100x+.
- **Antithetic Variates**: Use variance reduction techniques (simulate path and its negative) for Monte Carlo convergence.

## 📚 References

- **Stochastic Calculus for Finance**: Shreve
- **Options, Futures, and Other Derivatives**: Hull
- **Paul Wilmott on Quantitative Finance**

---

*Master stochastic processes to understand the random nature of financial markets!*
