# Advanced Python – Error Handling

## 📋 Overview

Robust error handling is what separates a script that crashes overnight from a professional trading system that runs for years. This module teaches you how to anticipate, catch, and manage errors gracefully.

## 🎯 Key Concepts

### **Try / Except / Else / Finally**
- **Try**: Run potentially risky code
- **Except**: Catch specific errors (e.g., `ZeroDivisionError`, `ValueError`)
- **Else**: Run if NO exception occurs
- **Finally**: Run ALWAYS (good for cleanup like closing connections)

### **Custom Exceptions**
- Create domain-specific errors (e.g., `InsufficientFundsError`, `MarketClosedError`) to make your code more readable and easier to debug.

### **Context Managers (`with`)**
- Automatically manage resources (files, network connections) to ensure they are closed properly, even if errors occur.

### **Logging**
- Stop using `print()` for errors! Use the `logging` module to record timestamps, error levels (INFO, WARNING, ERROR), and stack traces.

## 💻 Key Examples

### Basic Error Handling
```python
try:
    price = get_price("AAPL")
    shares = 1000 / price
except ZeroDivisionError:
    print("Price cannot be zero!")
except ValueError:
    print("Invalid ticker symbol")
else:
    print(f"Bought {shares} shares")
finally:
    print("Trade attempt complete")
```

### Custom Exception
```python
class InsufficientFundsError(Exception):
    pass

def buy(amount):
    if amount > balance:
        raise InsufficientFundsError("Not enough cash!")
```

## 📂 Files
- `error_handling_tutorial.py`: Interactive tutorial with examples

## 🚀 How to Run
```bash
python error_handling_tutorial.py
```

## 🧠 Financial Applications

### 1. API Connection Failures
Handle network timeouts or rate limits when fetching market data. Use exponential backoff to retry.

### 2. Data Validation
Validate trade inputs (e.g., positive price, valid ticker) before sending orders to an exchange.

### 3. Order Execution
Handle partial fills or rejected orders gracefully without crashing the entire bot.

### 4. System Monitoring
Use logging to track every error in a file, so you can debug why a trade failed yesterday at 3 AM.

## 💡 Best Practices

- **Be Specific**: Catch `ValueError` instead of `Exception`.
- **Don't Swallow Errors**: Avoid `except: pass` unless you really mean it.
- **Fail Fast**: Validate inputs early.
- **Log Everything**: In finance, an unlogged error can cost money.

---

*Write code that survives the chaos of real markets!*
