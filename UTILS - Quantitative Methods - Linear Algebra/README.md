# Quantitative Methods – Linear Algebra

## 📋 Overview

Linear algebra is the mathematical foundation for portfolio optimization, risk modeling, factor analysis, and quantitative finance. This utility teaches essential concepts through practical financial applications.

## 🎯 Key Concepts

### **Vectors**
- **Portfolio weights**: Allocation across assets
- **Returns**: Expected returns as vectors
- **Dot product**: Portfolio expected return calculation
- **Vector norms**: Risk and distance metrics

### **Matrices**
- **Returns matrix**: Assets × time periods
- **Covariance matrix**: Asset risk relationships
- **Correlation matrix**: Normalized dependencies
- **Matrix operations**: Addition, multiplication, transpose

### **Covariance & Correlation**
- **Covariance**: Measure of joint variability
- **Correlation**: Normalized covariance (-1 to 1)
- **Portfolio variance**: σ²_p = w^T Σ w
- **Diversification benefit**: Reduce risk through low correlation

### **Eigenvalues & Eigenvectors**
- **Principal Component Analysis (PCA)**: Dimensionality reduction
- **Risk factors**: Identify major sources of portfolio risk
- **Variance explained**: How much variation each factor captures
- **Factor models**: Multi-factor risk attribution

### **Matrix Operations**
- **Inverse**: Solve for optimal portfolios
- **Determinant**: Check invertibility
- **Transpose**: Switch rows and columns
- **Solving systems**: Optimize with constraints

## 💻 Key Examples

### Portfolio Expected Return
```python
weights = np.array([0.3, 0.3, 0.4])
expected_returns = np.array([0.10, 0.12, 0.08])

# Dot product
portfolio_return = weights @ expected_returns  # 9.8%
```

### Portfolio Variance
```python
# σ²_portfolio = w^T Σ w
portfolio_variance = weights @ cov_matrix @ weights
portfolio_vol = np.sqrt(portfolio_variance)
```

### Minimum Variance Portfolio
```python
# w_mvp = (Σ^-1 1) / (1^T Σ^-1 1)
cov_inv = np.linalg.inv(cov_matrix)
ones = np.ones(n_assets)
weights_mvp = (cov_inv @ ones) / (ones @ cov_inv @ ones)
```

### Principal Component Analysis
```python
# Find major risk factors
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

# First eigenvalue = proportion of variance from market factor
market_factor_weight = eigenvalues[0] / sum(eigenvalues)
```

## 📂 Files
- `linear_algebra_tutorial.py`: Comprehensive tutorial with examples
- `README.md`: This file

## 🚀 How to Run
```bash
pip install numpy
python linear_algebra_tutorial.py
```

## 🧠 Financial Applications

### 1. Portfolio Optimization
Use matrix operations to find optimal portfolio weights:
- **Minimum variance portfolio**: Lowest risk
- **Maximum Sharpe portfolio**: Best risk-adjusted return
- **Efficient frontier**: Trade-off between risk and return

### 2. Risk Attribution
Decompose portfolio risk by asset/factor:
- **Marginal risk**: Change in risk from small weight change
- **Risk contribution**: Each asset's contribution to total variance
- **Factor exposure**: Sensitivity to market factors

### 3. Factor Models
Multi-factor risk models (Fama-French, etc.):
- **Market factor**: Overall market beta
- **Size factor**: Small-cap vs large-cap
- **Value factor**: Value vs growth stocks

### 4. Dimensionality Reduction
Simplify complex portfolios:
- **PCA**: Reduce 100+ stocks to 5-10 factors
- **Factor investing**: Exposures to systematic factors
- **Risk budgeting**: Allocate risk across factors

## 📚 Mathematical Foundations

### Portfolio Variance Formula
```
σ²_portfolio = Σᵢ Σⱼ wᵢ wⱼ σᵢⱼ

In matrix notation:
σ²_p = w^T Σ w

Where:
- w = vector of portfolio weights
- Σ = covariance matrix
- σᵢⱼ = covariance between assets i and j
```

### Eigendecomposition
```
Σ = Q Λ Q^T

Where:
- Σ = covariance matrix
- Q = matrix of eigenvectors
- Λ = diagonal matrix of eigenvalues
```

### Minimum Variance Portfolio
```
min w^T Σ w
subject to: w^T 1 = 1

Solution: w_mvp = (Σ^-1 1) / (1^T Σ^-1 1)
```

## 🎓 Practice Problems

1. **Equal Weight vs Minimum Variance**
   - Create a 5-asset portfolio with equal weights
   - Calculate the minimum variance portfolio
   - Compare volatilities

2. **Risk Contribution Analysis**
   - Calculate marginal risk for each asset
   - Compute risk contribution percentages
   - Identify which assets contribute most to risk

3. **Principal Component Analysis**
   - Perform PCA on a correlation matrix
   - Determine how many factors explain 90% of variance
   - Interpret the factor loadings

4. **Efficient Frontier**
   - Generate 1000 random portfolios
   - Calculate return and volatility for each
   - Plot efficient frontier

## 📖 References

- **Markowitz Portfolio Theory**: "Portfolio Selection" (1952)
- **Linear Algebra**: Gilbert Strang's textbook
- **NumPy Documentation**: https://numpy.org/doc/
- **Quantitative Portfolio Management**: Attilio Meucci

## 💡 Key Takeaways

✓ **Covariance matrix** is central to portfolio risk  
✓ **Matrix multiplication** computes portfolio metrics efficiently  
✓ **Eigenvalues** reveal dominant risk factors  
✓ **Matrix inverse** solves for optimal weights  
✓ **PCA** reduces dimensionality while preserving information  

---

*Master linear algebra to build sophisticated portfolio optimization and risk management systems!*
