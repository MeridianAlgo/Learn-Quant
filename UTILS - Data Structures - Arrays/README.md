# Arrays - Complete Guide to NumPy for Beginners and Beyond

## 📋 Overview

Welcome to the comprehensive guide to NumPy arrays! This utility is designed to help both beginners and experienced Python programmers master array operations for data analysis, scientific computing, and quantitative finance.

### 🎯 Who is this for?
- **Beginners** learning Python and numerical computing
- **Data Scientists** working with large datasets
- **Financial Analysts** performing quantitative analysis
- **Researchers** in scientific computing
- **Students** learning numerical methods

### 🚀 Why Use NumPy Arrays?
- **Speed**: Up to 100x faster than Python lists for numerical operations
- **Convenience**: Powerful built-in functions for mathematics, statistics, and linear algebra
- **Memory Efficiency**: Optimized storage for numerical data
- **Interoperability**: Works seamlessly with other scientific Python libraries
- **Versatility**: Handle multi-dimensional data with ease

## 🎓 Core Concepts

### **1. Understanding NumPy Arrays**

#### **What is a NumPy Array?**
A NumPy array is a grid of values, all of the same type, indexed by non-negative integers. It's the fundamental data structure in numerical computing with Python.

#### **Key Characteristics**
- **Homogeneous**: All elements must be of the same data type
- **Fixed Size**: The size cannot be changed after creation
- **Efficient**: Uses contiguous memory for better performance
- **Vectorized Operations**: Apply operations to entire arrays without loops

### **2. Array Types and Dimensions**

#### **1D Arrays (Vectors)**
```python
import numpy as np

# Create a 1D array from a list
prices = np.array([100, 101, 102, 103, 104])
print("1D Array:", prices)
print("Shape:", prices.shape)  # (5,)
print("Dimensions:", prices.ndim)  # 1
```

#### **2D Arrays (Matrices)**
```python
# Create a 2D array (matrix)
portfolio = np.array([
    [100, 200, 300],  # Stock A prices
    [50, 100, 150],   # Stock B prices
    [75, 150, 225]    # Stock C prices
])
print("\n2D Array:")
print(portfolio)
print("Shape:", portfolio.shape)  # (3, 3)
print("Dimensions:", portfolio.ndim)  # 2
```

#### **N-dimensional Arrays**
```python
# Create a 3D array
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("\n3D Array:")
print(tensor)
print("Shape:", tensor.shape)  # (2, 2, 2)
print("Dimensions:", tensor.ndim)  # 3
```

## 🛠️ Array Creation Methods

### **Basic Creation**
```python
# Create array of zeros
zeros = np.zeros(5)  # [0., 0., 0., 0., 0.]

# Create array of ones
ones = np.ones((2, 3))  # 2x3 array of ones

# Create identity matrix
identity = np.eye(3)  # 3x3 identity matrix

# Create array with range
range_array = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Create linearly spaced array
linspace = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]

# Create random array
random_array = np.random.rand(3, 3)  # 3x3 array of random numbers between 0 and 1
```

### **Special Arrays**
```python
# Diagonal matrix
diag = np.diag([1, 2, 3, 4])

# Upper triangular matrix
tri_upper = np.triu(np.ones((3, 3)))

# Lower triangular matrix
tri_lower = np.tril(np.ones((3, 3)))
```

## 🔄 Array Operations

### **Basic Operations**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
print("Addition:", a + b)        # [5 7 9]
print("Subtraction:", a - b)     # [-3 -3 -3]
print("Multiplication:", a * b)  # [4 10 18] (element-wise)
print("Division:", b / a)        # [4.  2.5 2. ]
print("Power:", a ** 2)         # [1 4 9]

# Matrix multiplication
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])
print("Matrix multiplication:")
print(np.matmul(matrix_a, matrix_b))
# [[19 22]
#  [43 50]]
```

### **Aggregation Functions**
```python
arr = np.array([1, 2, 3, 4, 5])

print("Sum:", np.sum(arr))        # 15
print("Mean:", np.mean(arr))      # 3.0
print("Standard Deviation:", np.std(arr))  # 1.414...
print("Min:", np.min(arr))        # 1
print("Max:", np.max(arr))        # 5
print("Index of max:", np.argmax(arr))  # 4
```

## 📊 Real-World Applications

### **Financial Analysis**
```python
# Calculate daily returns
prices = np.array([100, 102, 101, 103, 105, 104])
daily_returns = (prices[1:] - prices[:-1]) / prices[:-1]
print("Daily Returns (%):", daily_returns * 100)

# Calculate cumulative returns
cumulative_returns = (1 + daily_returns).cumprod() - 1
print("Cumulative Returns (%):", cumulative_returns[-1] * 100)
```

### **Portfolio Analysis**
```python
# Portfolio weights
weights = np.array([0.4, 0.3, 0.3])

# Expected returns
returns = np.array([0.08, 0.12, 0.15])

# Portfolio expected return
portfolio_return = np.dot(weights, returns)
print(f"Portfolio Expected Return: {portfolio_return*100:.2f}%")
```

## 🚀 Advanced Topics

### **Broadcasting**
```python
# Add a scalar to an array
a = np.array([1, 2, 3])
print(a + 5)  # [6, 7, 8]

# Add two arrays of different shapes
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(a + b)  # [[11, 22, 33], [14, 25, 36]]
```

### **Boolean Indexing**
```python
# Create boolean mask
data = np.array([1, 2, 3, 4, 5])
mask = data > 2
print("Mask:", mask)  # [False, False, True, True, True]

# Apply mask
filtered = data[mask]
print("Filtered:", filtered)  # [3, 4, 5]
```

## 📚 Additional Resources

### **Official Documentation**
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)

### **Recommended Learning**
- NumPy Tutorial on W3Schools
- Python Data Science Handbook by Jake VanderPlas
- NumPy exercises on Kaggle

### **Cheat Sheets**
- NumPy Cheat Sheet by DataCamp
- Python for Data Science Cheat Sheet

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### **Performance Benefits**
- **Vectorized Computing**: 10-100x faster than loops
- **Memory Efficiency**: Contiguous memory layout
- **Broadcasting**: Automatic dimension alignment
- **Mathematical Functions**: Built-in financial functions

## 💻 Implementation

### **Core Array Operations**
```python
import numpy as np

# Create arrays
prices = np.array([100, 101, 102, 103, 104])  # Price series
weights = np.array([0.4, 0.3, 0.3])           # Portfolio weights

# Vectorized calculations
returns = np.diff(prices) / prices[:-1]       # Return series
portfolio_return = np.dot(weights, returns)   # Portfolio return
```

### **Array Creation Methods**
```python
# From lists
arr1 = np.array([1, 2, 3, 4, 5])

# Zeros and ones
zeros = np.zeros((3, 3))      # 3x3 zero matrix
ones = np.ones((2, 5))        # 2x5 ones matrix

# Random arrays (for Monte Carlo)
random_returns = np.random.normal(0.08, 0.15, 1000)

# Sequences
time_series = np.arange(0, 100, 1)  # Time periods
```

### **Advanced Array Operations**
```python
# Covariance matrix calculation
returns_matrix = np.random.normal(0.001, 0.02, (100, 5))
cov_matrix = np.cov(returns_matrix.T)

# Matrix operations
identity = np.eye(3)          # Identity matrix
inverse_cov = np.linalg.inv(cov_matrix)

# Eigenvalue decomposition for risk analysis
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

## 📊 Examples

### **Example 1: Portfolio Risk Analysis**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample returns
np.random.seed(42)
n_assets = 5
n_periods = 252  # Trading days

returns = np.random.multivariate_normal(
    mean=[0.001, 0.0008, 0.0012, 0.0009, 0.0011],
    cov=np.array([
        [0.0004, 0.0002, 0.0001, 0.00015, 0.0001],
        [0.0002, 0.0003, 0.00015, 0.0001, 0.00012],
        [0.0001, 0.00015, 0.0005, 0.0002, 0.00018],
        [0.00015, 0.0001, 0.0002, 0.0004, 0.00016],
        [0.0001, 0.00012, 0.00018, 0.00016, 0.00035]
    ]),
    size=n_periods
)

# Calculate portfolio statistics
portfolio_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
portfolio_returns = np.dot(returns, portfolio_weights)

print(f"Portfolio Mean Return: {np.mean(portfolio_returns):.6f}")
print(f"Portfolio Volatility: {np.std(portfolio_returns):.6f}")
print(f"Sharpe Ratio: {np.mean(portfolio_returns) / np.std(portfolio_returns):.4f}")
```

### **Example 2: Time Series Analysis**
```python
# Simulate stock price paths using geometric Brownian motion
def simulate_gbm(s0, mu, sigma, t, n_paths, n_steps):
    dt = t / n_steps
    price_paths = np.zeros((n_paths, n_steps + 1))
    price_paths[:, 0] = s0

    for i in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        price_paths[:, i] = price_paths[:, i-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return price_paths

# Parameters
s0 = 100      # Initial price
mu = 0.08     # Expected return
sigma = 0.20  # Volatility
t = 1         # Time horizon (1 year)
n_paths = 1000
n_steps = 252

paths = simulate_gbm(s0, mu, sigma, t, n_paths, n_steps)

# Calculate statistics
final_prices = paths[:, -1]
print(f"Mean final price: ${np.mean(final_prices):.2f}")
print(f"Median final price: ${np.median(final_prices):.2f}")
print(f"95% VaR: ${np.percentile(final_prices, 5):.2f}")
```

### **Example 3: Linear Algebra for Finance**
```python
# Solve for optimal portfolio weights using matrix algebra
# Maximize Sharpe ratio: w'μ / sqrt(w'Σw) subject to w'1 = 1

def optimal_portfolio_weights(mu, cov_matrix):
    n = len(mu)
    ones = np.ones(n)

    # Lagrange multiplier solution
    A = np.vstack([np.zeros((1, n)), ones])
    A = np.vstack([A, np.column_stack([ones, cov_matrix])])

    b = np.zeros(n + 2)
    b[0] = 1  # Sharpe ratio maximization
    b[1] = 1  # Budget constraint

    solution = np.linalg.solve(A, b)
    weights = solution[2:]

    return weights

# Example usage
expected_returns = np.array([0.12, 0.08, 0.15, 0.10])
cov_matrix = np.array([
    [0.04, 0.02, 0.03, 0.025],
    [0.02, 0.03, 0.025, 0.02],
    [0.03, 0.025, 0.06, 0.04],
    [0.025, 0.02, 0.04, 0.035]
])

optimal_weights = optimal_portfolio_weights(expected_returns, cov_matrix)
print(f"Optimal weights: {optimal_weights}")
print(f"Sum of weights: {np.sum(optimal_weights):.6f}")
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python -m pytest tests/test_arrays.py -v
```

## 📚 References

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [Quantitative Finance with Python](https://www.quantconnect.com/docs/home2)
- [Linear Algebra for Finance](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

## 🎓 Learning Path

### **Prerequisites**
- Basic Python programming
- Understanding of financial returns and volatility

### **Next Steps**
- **DataFrames**: Apply array operations to tabular data
- **Matrices**: Advanced linear algebra for risk modeling
- **Statistics**: Statistical analysis using NumPy arrays

### **Assessment**
1. Create a function that calculates portfolio volatility given weights and covariance matrix
2. Implement a simple Monte Carlo simulation for option pricing
3. Build a factor model using matrix operations

---

*This utility is part of the comprehensive quantitative finance learning platform. Master arrays to unlock powerful numerical computing capabilities for financial analysis.*
