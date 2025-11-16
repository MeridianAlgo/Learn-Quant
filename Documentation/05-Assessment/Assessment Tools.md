# 🧪 Assessment Tools - Test Your Knowledge

## 🎯 Comprehensive Assessment System

This assessment system provides quizzes, exercises, and evaluation tools to test your understanding of quantitative finance concepts and measure your progress.

---

## 📋 Assessment Overview

### **Assessment Types**
- **Knowledge Quizzes** - Test theoretical understanding
- **Coding Exercises** - Practice implementation skills
- **Case Studies** - Apply concepts to real scenarios
- **Project Evaluations** - Assess complete applications
- **Performance Benchmarks** - Measure against standards

### **Assessment Levels**
- **Beginner** - Basic concepts and simple calculations
- **Intermediate** - Complex calculations and analysis
- **Advanced** - Advanced modeling and optimization
- **Expert** - Research-level problems and innovation

---

## 📝 Knowledge Quizzes

### **Quiz 1: Python Fundamentals**

#### **Multiple Choice Questions**
1. **What is the correct way to create a list in Python?**
   - a) `list = (1, 2, 3)`
   - b) `list = [1, 2, 3]`
   - c) `list = {1, 2, 3}`
   - d) `list = "1, 2, 3"`

2. **Which data type is immutable in Python?**
   - a) List
   - b) Dictionary
   - c) Tuple
   - d) Set

3. **What does the `len()` function return?**
   - a) The first element of a sequence
   - b) The number of elements in a sequence
   - c) The last element of a sequence
   - d) The type of a sequence

#### **Coding Questions**
1. **Write a function that calculates the average of a list of numbers.**
2. **Create a dictionary with stock tickers as keys and prices as values.**
3. **Write a list comprehension that squares all numbers in a list.**

#### **Answer Key**
1. **b) `list = [1, 2, 3]`**
2. **c) Tuple**
3. **b) The number of elements in a sequence**

---

### **Quiz 2: Time Value of Money**

#### **Multiple Choice Questions**
1. **What is the present value formula?**
   - a) `PV = FV × (1 + r)^n`
   - b) `PV = FV / (1 + r)^n`
   - c) `PV = FV × r × n`
   - d) `PV = FV + (FV × r × n)`

2. **If you invest $1,000 at 5% annual interest for 3 years, what is the future value?**
   - a) $1,050.00
   - b) $1,102.50
   - c) $1,150.00
   - d) $1,157.63

3. **What is the effective annual rate for 12% nominal rate with monthly compounding?**
   - a) 12.00%
   - b) 12.68%
   - c) 12.36%
   - d) 13.00%

#### **Calculation Questions**
1. **Calculate the monthly payment for a $300,000 mortgage at 4.5% for 30 years.**
2. **What is the NPV of $10,000 received each year for 5 years at a 6% discount rate?**
3. **Calculate the IRR for an investment of $100,000 that returns $30,000 annually for 5 years.**

#### **Answer Key**
1. **b) `PV = FV / (1 + r)^n`**
2. **d) $1,157.63**
3. **b) 12.68%**

---

### **Quiz 3: Statistical Analysis**

#### **Multiple Choice Questions**
1. **What does standard deviation measure?**
   - a) Central tendency
   - b) Data dispersion
   - c) Data frequency
   - d) Data correlation

2. **What is the Sharpe ratio formula?**
   - a) `(Return - Risk Free) / Volatility`
   - b) `(Return + Risk Free) / Volatility`
   - c) `Return / (Risk Free × Volatility)`
   - d) `Volatility / (Return - Risk Free)`

3. **What does a p-value less than 0.05 indicate?**
   - a) Strong evidence against the null hypothesis
   - b) Strong evidence for the null hypothesis
   - c) No evidence either way
   - d) Invalid test results

#### **Calculation Questions**
1. **Calculate the standard deviation of returns: [0.02, -0.01, 0.03, -0.02, 0.01]**
2. **Compute the Sharpe ratio for a fund with 12% return, 2% risk-free rate, and 15% volatility.**
3. **What is the 95% VaR for a portfolio with $1M value and 20% annual volatility?**

#### **Answer Key**
1. **b) Data dispersion**
2. **a) `(Return - Risk Free) / Volatility`**
3. **a) Strong evidence against the null hypothesis**

---

## 💻 Coding Exercises

### **Exercise 1: Portfolio Calculator**
**Difficulty**: Beginner
**Time**: 15 minutes

**Problem**: Create a function that calculates portfolio return and volatility given asset weights and returns.

```python
def calculate_portfolio_metrics(weights, returns_matrix):
    """
    Calculate portfolio return and volatility.

    Args:
        weights: List of asset weights (must sum to 1)
        returns_matrix: 2D array of asset returns

    Returns:
        dict with 'return' and 'volatility' keys
    """
    # Your code here
    pass
```

**Test Case**:
```python
weights = [0.4, 0.3, 0.3]
returns = np.array([
    [0.01, 0.02, -0.01, 0.03],
    [0.02, -0.01, 0.02, 0.01],
    [-0.01, 0.03, 0.01, -0.02]
])

result = calculate_portfolio_metrics(weights, returns)
# Expected: {'return': 0.01, 'volatility': 0.015}
```

### **Exercise 2: Options Pricer**
**Difficulty**: Intermediate
**Time**: 30 minutes

**Problem**: Implement Black-Scholes option pricing with Greeks calculation.

```python
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price and Greeks.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'

    Returns:
        dict with 'price', 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    # Your code here
    pass
```

**Test Case**:
```python
result = black_scholes_greeks(100, 100, 1, 0.05, 0.2, 'call')
# Expected: {'price': 10.45, 'delta': 0.637, 'gamma': 0.019, 'vega': 37.52, 'theta': -6.41, 'rho': 53.75}
```

### **Exercise 3: Risk Analysis System**
**Difficulty**: Advanced
**Time**: 45 minutes

**Problem**: Build a comprehensive risk analysis system for a multi-asset portfolio.

```python
def comprehensive_risk_analysis(returns_matrix, confidence_level=0.95):
    """
    Perform comprehensive risk analysis.

    Args:
        returns_matrix: 2D array of asset returns
        confidence_level: Confidence level for VaR calculation

    Returns:
        dict with risk metrics
    """
    # Your code here
    pass
```

**Test Case**:
```python
returns = np.random.normal(0.001, 0.02, (100, 5))  # 100 days, 5 assets
result = comprehensive_risk_analysis(returns)
# Expected: Should include VaR, CVaR, volatility, correlations, etc.
```

---

## 📊 Case Studies

### **Case Study 1: Investment Analysis**

#### **Scenario**
You are a financial analyst evaluating Apple Inc. (AAPL) for investment. You have the following data:
- Current stock price: $150.25
- Expected annual return: 12%
- Volatility: 25%
- Risk-free rate: 2%
- Market risk premium: 6%
- Beta: 1.2

#### **Questions**
1. **Calculate the required return using CAPM.**
2. **Value the stock using dividend discount model (assuming 2% dividend growth).**
3. **Calculate Value at Risk for a $100,000 position.**
4. **Assess the investment attractiveness.**

#### **Solution Guide**
```python
# CAPM Calculation
required_return = risk_free + beta * market_premium
# = 0.02 + 1.2 * 0.06 = 0.092 (9.2%)

# DDM Valuation (simplified)
dividend_yield = 0.006  # 0.6%
growth_rate = 0.02  # 2%
ddm_value = (150.25 * dividend_yield) / (required_return - growth_rate)
# = (150.25 * 0.006) / (0.092 - 0.02) = 0.9015 / 0.072 = $12.52

# This is significantly below current price, suggesting overvaluation
```

### **Case Study 2: Portfolio Optimization**

#### **Scenario**
You manage a $1M portfolio with the following assets:
- Large Cap Stocks: Expected return 10%, volatility 18%
- Small Cap Stocks: Expected return 14%, volatility 25%
- Bonds: Expected return 4%, volatility 6%
- International Stocks: Expected return 11%, volatility 22%

Correlation matrix:
```
    Large    Small    Bonds    Intl
Large  1.00    0.75    0.10    0.65
Small  0.75    1.00    0.05    0.70
Bonds  0.10    0.05    1.00    0.15
Intl   0.65    0.70    0.15    1.00
```

#### **Questions**
1. **Find the optimal portfolio weights for maximum Sharpe ratio.**
2. **Calculate the efficient frontier.**
3. **Determine the minimum variance portfolio.**
4. **Recommend portfolio allocation for a moderate risk investor.**

#### **Solution Approach**
1. **Use mean-variance optimization**
2. **Implement Monte Carlo simulation for efficient frontier**
3. **Calculate portfolio statistics for different weight combinations**
4. **Select optimal allocation based on risk tolerance**

---

## 🏆 Performance Benchmarks

### **Coding Performance Standards**

#### **Beginner Level**
- **Syntax Accuracy**: 95%+ correct Python syntax
- **Code Organization**: Basic functions and comments
- **Problem Solving**: Solve simple algorithmic problems
- **Time Complexity**: Understand basic O(n) vs O(n²) concepts

#### **Intermediate Level**
- **Code Efficiency**: Use appropriate data structures
- **Error Handling**: Comprehensive try-catch blocks
- **Documentation**: Complete docstrings and comments
- **Testing**: Basic unit tests for functions

#### **Advanced Level**
- **Algorithm Optimization**: Implement efficient algorithms
- **System Design**: Object-oriented design patterns
- **Performance**: Meet production-level performance standards
- **Scalability**: Handle large datasets efficiently

### **Financial Knowledge Standards**

#### **TVM Mastery**
- Calculate PV/FV for complex cash flows
- Understand different compounding methods
- Apply to real-world investment decisions

#### **Statistical Proficiency**
- Calculate and interpret risk metrics
- Perform hypothesis testing
- Understand probability distributions

#### **Portfolio Management**
- Implement mean-variance optimization
- Calculate performance attribution
- Understand risk management principles

---

## 📈 Progress Tracking

### **Self-Assessment Checklist**

#### **Python Fundamentals**
- [ ] Basic syntax and data types
- [ ] Control flow (if/else, loops)
- [ ] Functions and modules
- [ ] Data structures (lists, dicts, sets)
- [ ] File I/O operations
- [ ] Error handling

#### **Numerical Computing**
- [ ] NumPy array operations
- [ ] Matrix calculations
- [ ] Random number generation
- [ ] Performance optimization
- [ ] Data visualization

#### **Financial Calculations**
- [ ] Time value of money
- [ ] Statistical analysis
- [ ] Risk metrics (VaR, Sharpe)
- [ ] Portfolio optimization
- [ ] Options pricing

#### **Advanced Topics**
- [ ] Machine learning basics
- [ ] Database operations
- [ ] API integration
- [ ] Web development
- [ ] System architecture

### **Skill Level Definitions**

#### **Beginner (Score: 0-25)**
- Basic understanding of concepts
- Can implement simple calculations
- Limited problem-solving ability
- Needs guidance for most tasks

#### **Intermediate (Score: 26-50)**
- Solid understanding of core concepts
- Can build functional applications
- Independent problem-solving
- Some advanced topic knowledge

#### **Advanced (Score: 51-75)**
- Deep understanding of concepts
- Can design complex systems
- Expert-level problem-solving
- Advanced topic proficiency

#### **Expert (Score: 76-100)**
- Mastery of all concepts
- Can innovate and research
- Teaching and mentoring ability
- Industry-level expertise

---

## 🎯 Certification Preparation

### **CFA Level 1 Preparation**
- **Quantitative Methods**: TVM, statistics, probability
- **Financial Reporting**: Statement analysis, ratios
- **Equity Investments**: Valuation, market efficiency
- **Fixed Income**: Bond pricing, yield calculations
- **Portfolio Management**: MPT, CAPM, optimization

### **FRM Part 1 Preparation**
- **Risk Management Foundations**: Probability, statistics
- **Quantitative Analysis**: Models, estimation
- **Financial Markets**: Products, pricing
- **Valuation**: Derivatives, fixed income

### **Python Certifications**
- **PCEP (Certified Entry-Level Python Programmer)**
- **PCAP (Certified Associate in Python Programming)**
- **PCPP (Certified Professional in Python Programming)**

---

## 📋 Final Assessment

### **Comprehensive Evaluation**

#### **Technical Skills Test**
1. **Python Proficiency** - Implement a complete financial application
2. **Data Analysis** - Analyze a financial dataset
3. **Algorithm Implementation** - Solve a complex algorithmic problem
4. **System Design** - Design a multi-component financial system

#### **Financial Knowledge Test**
1. **TVM and Valuation** - Complex financial calculations
2. **Risk Management** - Risk analysis and modeling
3. **Portfolio Theory** - Optimization and attribution
4. **Investment Analysis** - Company and market analysis

#### **Project Presentation**
1. **Code Quality** - Clean, documented, efficient code
2. **Problem Solving** - Innovative solutions to problems
3. **Business Understanding** - Clear explanation of financial concepts
4. **Communication** - Present findings effectively

### **Scoring Rubric**
- **Technical Implementation**: 40%
- **Financial Knowledge**: 30%
- **Problem Solving**: 20%
- **Communication**: 10%

### **Assessment Results**
- **Score 90-100**: Exceptional - Ready for senior roles
- **Score 80-89**: Excellent - Ready for professional roles
- **Score 70-79**: Good - Ready for junior roles
- **Score 60-69**: Satisfactory - Needs more practice
- **Score <60**: Needs improvement - Review fundamentals

---

*This assessment system provides comprehensive evaluation tools to measure your progress and identify areas for improvement. Use these tools regularly to track your learning journey and prepare for professional certifications.*

**Made with ❤️ by MeridianAlgo**
