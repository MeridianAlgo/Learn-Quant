# 🐍 Python Fundamentals Tutorial

## 🎯 Complete Guide to Python for Quantitative Finance

This comprehensive tutorial covers all Python fundamentals needed for quantitative finance applications, from basic syntax to advanced data structures.

---

## 📋 Tutorial Overview

**Duration**: 2-3 hours
**Prerequisites**: None
**Difficulty Level**: Beginner
**Key Topics**: Python syntax, data types, control flow, functions, modules

---

## 🏁 Getting Started

### **Installing Python**

#### **Option 1: Official Python Installation**
1. Visit [python.org/downloads](https://python.org/downloads)
2. Download the latest Python 3.x version
3. Run the installer and follow the setup wizard
4. **Important**: Check "Add Python to PATH" during installation

#### **Option 2: Using Anaconda (Recommended for Data Science)**
1. Download Anaconda from [anaconda.com](https://anaconda.com)
2. Install Anaconda following the setup instructions
3. Anaconda includes Python, Jupyter, and most data science packages

#### **Verifying Installation**
Open a terminal or command prompt and run:
```bash
python --version
pip --version
```

You should see Python 3.x and pip version information.

---

## 🔤 Basic Python Syntax

### **Hello World - Your First Program**

Create a file called `hello.py` and add this code:

```python
# This is a comment - Python ignores these lines
print("Hello, Quantitative Finance World!")

# Variables - storing data
company_name = "Apple Inc."
stock_price = 150.25
shares_owned = 100

# Calculations
total_value = stock_price * shares_owned
print(f"Total value of {company_name}: ${total_value".2f"}")

# Multiple assignments
high, low, close = 155.50, 148.75, 150.25
print(f"Daily range: {high - low".2f"}")
```

Run the program:
```bash
python hello.py
```

### **Data Types**

Python has several built-in data types:

```python
# Numbers
integer_num = 42
float_num = 3.14159
complex_num = 3 + 4j

# Strings
company = "Apple Inc."
ticker = 'AAPL'
description = """This is a
multi-line string"""

# Boolean
is_profitable = True
is_volatile = False

# None type
no_value = None

# Type checking
print(type(stock_price))  # <class 'float'>
print(isinstance(stock_price, float))  # True
```

### **Variables and Naming Conventions**

```python
# Good variable names
stock_price = 150.25
company_name = "Apple Inc."
portfolio_value = 1000000
is_dividend_stock = True

# Constants (use uppercase)
PI = 3.14159
MAX_PORTFOLIO_SIZE = 1000000

# Don't use these names (they're Python keywords)
# if, for, while, def, class, import, return, etc.
```

---

## 🔀 Control Flow

### **Conditional Statements**

```python
# Basic if statement
stock_price = 150.25
if stock_price > 100:
    print("Expensive stock")
elif stock_price > 50:
    print("Moderate price")
else:
    print("Cheap stock")

# Multiple conditions
market_cap = 2.5e12  # 2.5 trillion
if stock_price > 100 and market_cap > 1e12:
    print("Large cap, expensive stock")
elif stock_price > 100 or market_cap > 1e12:
    print("Either expensive or large cap")

# Ternary operator
status = "Expensive" if stock_price > 100 else "Cheap"
print(f"Stock status: {status}")
```

### **Loops**

```python
# For loop with range
print("Counting to 5:")
for i in range(5):
    print(i)

print("\nCounting with step:")
for i in range(0, 10, 2):
    print(i)

# For loop with list
portfolio = ["AAPL", "GOOGL", "MSFT", "AMZN"]
print("\nPortfolio holdings:")
for stock in portfolio:
    print(f"- {stock}")

# While loop
print("\nCompound interest calculation:")
principal = 1000
rate = 0.05
years = 0
while principal < 2000 and years < 20:
    principal *= (1 + rate)
    years += 1
    print(f"Year {years}: ${principal".2f"}")
```

### **Loop Control**

```python
# Break and continue
print("Finding first expensive stock:")
prices = [50, 75, 125, 200, 90, 150]
for price in prices:
    if price > 100:
        print(f"Found expensive stock: ${price}")
        break

print("\nSkipping cheap stocks:")
for price in prices:
    if price < 100:
        continue
    print(f"Expensive stock: ${price}")

# Enumerate for index and value
print("\nPortfolio with indices:")
portfolio = ["AAPL", "GOOGL", "MSFT"]
for index, stock in enumerate(portfolio):
    print(f"{index + 1}. {stock}")
```

---

## 🧮 Functions

### **Defining Functions**

```python
# Basic function
def calculate_return(initial_price, final_price):
    """Calculate investment return percentage."""
    return (final_price - initial_price) / initial_price * 100

# Function with default parameters
def compound_interest(principal, rate, years=1, compounding=1):
    """Calculate compound interest."""
    return principal * (1 + rate / compounding) ** (years * compounding)

# Function with multiple return values
def analyze_stock(price, high, low):
    """Analyze stock price data."""
    daily_range = high - low
    volatility = daily_range / price * 100
    midpoint = (high + low) / 2
    return daily_range, volatility, midpoint

# Calling functions
aapl_return = calculate_return(150, 165)
print(f"AAPL return: {aapl_return".2f"}%")

savings = compound_interest(1000, 0.05, 10)
print(f"Future value: ${savings".2f"}")

range_data = analyze_stock(150, 155, 145)
print(f"Daily range: ${range_data[0]".2f"}")
print(f"Volatility: {range_data[1]".2f"}%")
print(f"Midpoint: ${range_data[2]".2f"}")
```

### **Scope and Global Variables**

```python
# Global variable
portfolio_value = 100000

def add_investment(amount):
    """Add investment to portfolio."""
    global portfolio_value
    portfolio_value += amount
    return portfolio_value

def calculate_portfolio_return(initial, final):
    """Calculate portfolio return (local scope)."""
    portfolio_return = (final - initial) / initial * 100
    # portfolio_value is not accessible here
    return portfolio_return

# Using the functions
print(f"Initial portfolio: ${portfolio_value",.2f"}")
new_value = add_investment(5000)
print(f"After investment: ${new_value",.2f"}")

portfolio_return = calculate_portfolio_return(100000, 110000)
print(f"Portfolio return: {portfolio_return".2f"}%")
```

### **Lambda Functions**

```python
# Simple lambda functions
square = lambda x: x ** 2
add_numbers = lambda x, y: x + y
is_positive = lambda x: x > 0

# Using lambdas
print(f"5 squared: {square(5)}")
print(f"10 + 15: {add_numbers(10, 15)}")
print(f"Is 5 positive? {is_positive(5)}")

# Lambda with financial calculations
calculate_yield = lambda principal, coupon, price: (coupon / price) * 100
bond_yield = calculate_yield(1000, 50, 950)
print(f"Bond yield: {bond_yield".2f"}%")

# Lambda in sorting
stocks = [
    {"ticker": "AAPL", "price": 150},
    {"ticker": "GOOGL", "price": 2800},
    {"ticker": "MSFT", "price": 350}
]

# Sort by price (ascending)
sorted_asc = sorted(stocks, key=lambda x: x["price"])
print("Sorted ascending:", [s["ticker"] for s in sorted_asc])

# Sort by price (descending)
sorted_desc = sorted(stocks, key=lambda x: x["price"], reverse=True)
print("Sorted descending:", [s["ticker"] for s in sorted_desc])
```

---

## 📦 Modules and Packages

### **Importing Modules**

```python
# Standard library imports
import math
import random
import datetime
import os

# Third-party imports (install with pip)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import specific functions
from math import sqrt, pi
from datetime import datetime as dt

# Import with alias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Using imported functions
print(f"Square root of 16: {sqrt(16)}")
print(f"Pi value: {pi".6f"}")
print(f"Random number: {random.random()}")

# NumPy operations
prices = np.array([100, 102, 98, 105, 107])
returns = np.diff(prices) / prices[:-1]
print(f"Average return: {np.mean(returns)".4f"}")
```

### **Creating Your Own Modules**

Create a file called `financial_utils.py`:

```python
# financial_utils.py
def calculate_simple_interest(principal, rate, time):
    """Calculate simple interest."""
    return principal * rate * time

def calculate_compound_interest(principal, rate, time, compounding=1):
    """Calculate compound interest."""
    return principal * (1 + rate / compounding) ** (time * compounding)

def calculate_return(initial_price, final_price):
    """Calculate investment return."""
    return (final_price - initial_price) / initial_price * 100

# Constants
INFLATION_RATE = 0.02
RISK_FREE_RATE = 0.03
```

Now use it in another file:

```python
# main.py
import financial_utils as fu

# Using the module
principal = 1000
rate = 0.05
time = 5

simple_interest = fu.calculate_simple_interest(principal, rate, time)
compound_interest = fu.calculate_compound_interest(principal, rate, time)

print(f"Simple interest: ${simple_interest".2f"}")
print(f"Compound interest: ${compound_interest".2f"}")

# Using constants
print(f"Inflation rate: {fu.INFLATION_RATE".2%"}")
```

### **Installing External Packages**

```python
# Install packages using pip
pip install numpy
pip install pandas
pip install matplotlib
pip install scipy
pip install scikit-learn

# Install specific versions
pip install numpy==1.21.0
pip install pandas>=1.3.0

# Install from requirements file
pip install -r requirements.txt

# Upgrade packages
pip install --upgrade numpy pandas

# List installed packages
pip list

# Save current environment
pip freeze > requirements.txt
```

---

## 🗂️ Data Structures

### **Lists - Ordered Collections**

```python
# Creating lists
empty_list = []
stock_prices = [150.25, 152.50, 148.75, 155.00]
mixed_list = ["AAPL", 150.25, True, None]

# List operations
print(f"Length: {len(stock_prices)}")
print(f"First price: {stock_prices[0]}")
print(f"Last price: {stock_prices[-1]}")

# Adding elements
stock_prices.append(157.25)  # Add to end
stock_prices.insert(0, 149.00)  # Insert at index
stock_prices.extend([158.50, 160.00])  # Add multiple

# Removing elements
stock_prices.remove(149.00)  # Remove specific value
popped_price = stock_prices.pop()  # Remove and return last
del stock_prices[0]  # Remove by index

# List slicing
recent_prices = stock_prices[-5:]  # Last 5 prices
first_three = stock_prices[:3]  # First 3 prices
middle_prices = stock_prices[2:5]  # Prices 2-4

# List comprehensions
squared_prices = [price ** 2 for price in stock_prices]
high_prices = [price for price in stock_prices if price > 150]
price_changes = [stock_prices[i] - stock_prices[i-1] for i in range(1, len(stock_prices))]
```

### **Dictionaries - Key-Value Pairs**

```python
# Creating dictionaries
empty_dict = {}
stock_info = {
    "AAPL": 150.25,
    "GOOGL": 2800.50,
    "MSFT": 350.75
}

company_data = {
    "Apple Inc.": {
        "ticker": "AAPL",
        "price": 150.25,
        "sector": "Technology",
        "employees": 147000
    }
}

# Dictionary operations
print(f"AAPL price: {stock_info['AAPL']}")
stock_info["TSLA"] = 850.25  # Add new entry
stock_info["AAPL"] = 152.50  # Update existing

# Safe access
aapl_price = stock_info.get("AAPL", 0)  # Returns 152.50
unknown_price = stock_info.get("UNKNOWN", 0)  # Returns 0

# Dictionary methods
print(f"Keys: {list(stock_info.keys())}")
print(f"Values: {list(stock_info.values())}")
print(f"Items: {list(stock_info.items())}")

# Dictionary comprehensions
price_dict = {ticker: price * 1.01 for ticker, price in stock_info.items()}
high_value_stocks = {k: v for k, v in stock_info.items() if v > 500}

# Nested dictionaries
portfolio = {
    "account_001": {
        "holdings": {
            "AAPL": {"shares": 100, "avg_cost": 145.50},
            "GOOGL": {"shares": 10, "avg_cost": 2750.00}
        },
        "cash": 50000
    }
}

# Accessing nested data
aapl_shares = portfolio["account_001"]["holdings"]["AAPL"]["shares"]
```

### **Sets - Unique Collections**

```python
# Creating sets
empty_set = set()
unique_tickers = {"AAPL", "GOOGL", "MSFT", "AMZN"}
tech_sectors = {"Technology", "Healthcare", "Financial"}

# Set operations
print(f"Union: {unique_tickers | tech_sectors}")
print(f"Intersection: {unique_tickers & tech_sectors}")
print(f"Difference: {unique_tickers - tech_sectors}")

# Adding and removing
tech_sectors.add("Energy")
tech_sectors.remove("Healthcare")
tech_sectors.discard("NonExistent")  # Safe remove

# Set comprehensions
squares_set = {x**2 for x in range(10)}
even_squares = {x**2 for x in range(10) if x % 2 == 0}

# Practical financial example
portfolio_tickers = ["AAPL", "GOOGL", "AAPL", "MSFT", "GOOGL", "TSLA"]
unique_holdings = set(portfolio_tickers)
print(f"Unique holdings: {unique_holdings}")
```

### **Tuples - Immutable Sequences**

```python
# Creating tuples
empty_tuple = ()
single_item = (42,)  # Note the comma
stock_data = ("AAPL", 150.25, 155.50, 148.75)

# Tuple operations
ticker, open_price, high, low = stock_data
print(f"Stock: {ticker}, Open: ${open_price}")

# Tuple methods
print(f"Length: {len(stock_data)}")
print(f"Count of 150.25: {stock_data.count(150.25)}")
print(f"Index of 'AAPL': {stock_data.index('AAPL')}")

# Tuple unpacking
coordinates = (40.7128, -74.0060)  # NYC coordinates
lat, lon = coordinates

# Named tuples (from collections module)
from collections import namedtuple
Stock = namedtuple('Stock', ['ticker', 'price', 'volume'])
aapl = Stock('AAPL', 150.25, 50000000)
print(f"{aapl.ticker}: ${aapl.price}, Volume: {aapl.volume}")
```

---

## 🔧 Error Handling

### **Try-Except Blocks**

```python
def safe_division(dividend, divisor):
    """Safely divide two numbers."""
    try:
        result = dividend / divisor
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Invalid data types!")
        return None

# Using the function
result1 = safe_division(10, 2)  # Works fine
result2 = safe_division(10, 0)  # Handles error
result3 = safe_division("10", 2)  # Handles error

# Multiple exception handling
def parse_price(price_str):
    """Parse price string to float."""
    try:
        price = float(price_str)
        if price < 0:
            raise ValueError("Price cannot be negative")
        return price
    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except TypeError:
        print("Type error: Invalid input type")
        return None

# Finally block
def read_stock_data(filename):
    """Read stock data from file."""
    try:
        with open(filename, 'r') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    finally:
        print("File operation completed")
```

### **Custom Exceptions**

```python
class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds."""
    pass

class InvalidTickerError(Exception):
    """Raised when ticker symbol is invalid."""
    pass

def withdraw_funds(account_balance, amount):
    """Withdraw funds from account."""
    if amount > account_balance:
        raise InsufficientFundsError(f"Insufficient funds: {account_balance} < {amount}")
    return account_balance - amount

def validate_ticker(ticker):
    """Validate stock ticker symbol."""
    if not ticker.isalpha() or len(ticker) > 5:
        raise InvalidTickerError(f"Invalid ticker: {ticker}")
    return ticker.upper()

# Using custom exceptions
try:
    new_balance = withdraw_funds(1000, 1500)
except InsufficientFundsError as e:
    print(f"Transaction failed: {e}")

try:
    valid_ticker = validate_ticker("aapl123")
except InvalidTickerError as e:
    print(f"Validation failed: {e}")
```

---

## 📝 File I/O Operations

### **Reading and Writing Text Files**

```python
# Writing to a file
def save_portfolio(portfolio, filename):
    """Save portfolio data to file."""
    with open(filename, 'w') as file:
        file.write("Portfolio Holdings\n")
        file.write("=" * 20 + "\n")
        for ticker, data in portfolio.items():
            file.write(f"{ticker}: {data['shares']} shares @ ${data['price']".2f"}\n")

# Reading from a file
def load_portfolio(filename):
    """Load portfolio data from file."""
    portfolio = {}
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Skip header lines
        data_lines = [line.strip() for line in lines if ':' in line]

        for line in data_lines:
            ticker, info = line.split(':')
            shares_str, price_str = info.split(' shares @ $')
            portfolio[ticker.strip()] = {
                'shares': int(shares_str),
                'price': float(price_str)
            }
    except FileNotFoundError:
        print(f"File {filename} not found")
    except Exception as e:
        print(f"Error reading file: {e}")

    return portfolio

# Using the functions
portfolio = {
    "AAPL": {"shares": 100, "price": 150.25},
    "GOOGL": {"shares": 10, "price": 2800.50}
}

save_portfolio(portfolio, "my_portfolio.txt")
loaded_portfolio = load_portfolio("my_portfolio.txt")
print("Loaded portfolio:", loaded_portfolio)
```

### **Working with CSV Files**

```python
import csv

def save_portfolio_csv(portfolio, filename):
    """Save portfolio to CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ticker', 'Shares', 'Price', 'Value'])

        for ticker, data in portfolio.items():
            value = data['shares'] * data['price']
            writer.writerow([ticker, data['shares'], data['price'], value])

def load_portfolio_csv(filename):
    """Load portfolio from CSV file."""
    portfolio = {}
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                portfolio[row['Ticker']] = {
                    'shares': int(row['Shares']),
                    'price': float(row['Price']),
                    'value': float(row['Value'])
                }
    except FileNotFoundError:
        print(f"File {filename} not found")
    except Exception as e:
        print(f"Error reading CSV: {e}")

    return portfolio

# Using CSV functions
portfolio = {
    "AAPL": {"shares": 100, "price": 150.25},
    "GOOGL": {"shares": 10, "price": 2800.50},
    "MSFT": {"shares": 50, "price": 350.75}
}

save_portfolio_csv(portfolio, "portfolio.csv")
loaded_portfolio = load_portfolio_csv("portfolio.csv")
print("CSV Portfolio:", loaded_portfolio)
```

### **Working with JSON Files**

```python
import json

def save_portfolio_json(portfolio, filename):
    """Save portfolio to JSON file."""
    with open(filename, 'w') as file:
        json.dump(portfolio, file, indent=2)

def load_portfolio_json(filename):
    """Load portfolio from JSON file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in {filename}")
        return {}

# Using JSON functions
portfolio = {
    "AAPL": {
        "shares": 100,
        "price": 150.25,
        "sector": "Technology",
        "last_updated": "2024-01-15"
    },
    "GOOGL": {
        "shares": 10,
        "price": 2800.50,
        "sector": "Technology",
        "last_updated": "2024-01-15"
    }
}

save_portfolio_json(portfolio, "portfolio.json")
loaded_portfolio = load_portfolio_json("portfolio.json")
print("JSON Portfolio:", loaded_portfolio)
```

---

## 🧪 Testing Your Code

### **Basic Testing**

```python
def test_calculations():
    """Test financial calculations."""
    # Test simple interest
    result = calculate_simple_interest(1000, 0.05, 1)
    expected = 50.0
    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"

    # Test compound interest
    result = calculate_compound_interest(1000, 0.05, 1)
    expected = 1050.0
    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"

    # Test return calculation
    result = calculate_return(100, 110)
    expected = 10.0
    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"

    print("All tests passed!")

# Run tests
test_calculations()
```

### **Unit Testing with unittest**

```python
import unittest

class TestFinancialCalculations(unittest.TestCase):

    def test_simple_interest(self):
        """Test simple interest calculation."""
        result = calculate_simple_interest(1000, 0.05, 1)
        self.assertAlmostEqual(result, 50.0, places=2)

    def test_compound_interest(self):
        """Test compound interest calculation."""
        result = calculate_compound_interest(1000, 0.05, 1)
        self.assertAlmostEqual(result, 1050.0, places=2)

    def test_return_calculation(self):
        """Test return calculation."""
        result = calculate_return(100, 110)
        self.assertAlmostEqual(result, 10.0, places=2)

    def test_negative_inputs(self):
        """Test handling of negative inputs."""
        with self.assertRaises(ValueError):
            calculate_simple_interest(-1000, 0.05, 1)

# Run the tests
if __name__ == '__main__':
    unittest.main()
```

---

## 🎯 Practice Exercises

### **Exercise 1: Portfolio Tracker**
Build a simple portfolio tracker that:
- Stores stock holdings in a list of dictionaries
- Calculates total portfolio value
- Calculates individual position values
- Displays portfolio summary

### **Exercise 2: Financial Calculator**
Create a financial calculator that:
- Calculates simple and compound interest
- Computes loan payments
- Calculates investment returns
- Handles different compounding frequencies

### **Exercise 3: Data Analyzer**
Build a data analyzer that:
- Reads financial data from a CSV file
- Calculates basic statistics (mean, median, std)
- Identifies outliers and trends
- Generates summary reports

### **Exercise 4: Error Handling**
Enhance your financial calculator with:
- Comprehensive error handling
- Input validation
- Custom exception classes
- User-friendly error messages

---

## 📚 Next Steps

### **Continue Learning**
- **Data Structures**: Master lists, dictionaries, sets, and tuples
- **NumPy**: Learn numerical computing for finance
- **pandas**: Master data manipulation and analysis
- **Object-Oriented Programming**: Learn classes and objects

### **Recommended Practice**
- **LeetCode**: Solve Python coding problems
- **HackerRank**: Complete Python challenges
- **Project Euler**: Solve mathematical problems with Python
- **Kaggle**: Participate in data science competitions

### **Build Real Projects**
- **Personal Finance Tracker**: Track your expenses and budget
- **Stock Portfolio Manager**: Manage your investment portfolio
- **Retirement Calculator**: Plan for your financial future
- **Loan Comparison Tool**: Compare different loan options

---

*This tutorial provides a solid foundation in Python programming for quantitative finance. Practice these concepts regularly, and you'll be ready to tackle more advanced topics like data analysis, financial modeling, and algorithmic trading.*

**Made with ❤️ by MeridianAlgo**
