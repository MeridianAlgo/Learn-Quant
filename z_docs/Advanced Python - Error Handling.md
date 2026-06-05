<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Advanced Python</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Advanced Python - Error Handling"
    python "error_handling_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Advanced%20Python%20-%20Error%20Handling)

---
# Advanced Python – Error Handling

## Overview

Robust error handling is what separates a script that crashes overnight from a professional trading system that runs for years. This module teaches you how to anticipate, catch, and manage errors gracefully.

## Key Concepts

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

## Key Examples

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

## Files
- `error_handling_tutorial.py`: Interactive tutorial with examples

## How to Run
```bash
python error_handling_tutorial.py
```

## Financial Applications

### 1. API Connection Failures
Handle network timeouts or rate limits when fetching market data. Use exponential backoff to retry.

### 2. Data Validation
Validate trade inputs (e.g., positive price, valid ticker) before sending orders to an exchange.

### 3. Order Execution
Handle partial fills or rejected orders gracefully without crashing the entire bot.

### 4. System Monitoring
Use logging to track every error in a file, so you can debug why a trade failed yesterday at 3 AM.

## Best Practices

- **Be Specific**: Catch `ValueError` instead of `Exception`.
- **Don't Swallow Errors**: Avoid `except: pass` unless you really mean it.
- **Fail Fast**: Validate inputs early.
- **Log Everything**: In finance, an unlogged error can cost money.

---

*Write code that survives the chaos of real markets!*

---

## Continue in Advanced Python

<div class="grid cards" markdown>

-   :material-cog-outline: __[Advanced Python - AsyncIO](Advanced Python - AsyncIO.md)__

    In quantitative finance, speed is edge. Python's `asyncio` library allows for **concurrency**, letting your program handle multiple tasks (like fetching data from 10 different exchanges) at once, rather than waiting for one to finish before starting the next.

-   :material-cog-outline: __[Advanced Python - Context Managers](Advanced Python - Context Managers.md)__

    Context Managers are a powerful Python feature for resource management. They allow you to allocate and release resources precisely when you want to. The most common usage is the `with` statement.

-   :material-cog-outline: __[Advanced Python - Decorators and Generators](Advanced Python - Decorators and Generators.md)__

    Decorators and Generators are powerful Python features that separate professional code from beginner scripts. Decorators allow you to modify function behavior cleanly, while Generators enable memory-efficient processing of large financial datasets.

-   :material-cog-outline: __[Advanced Python - Multiprocessing](Advanced Python - Multiprocessing.md)__

    Python Global Interpreter Lock prevents multiple threads from executing Python bytecode at the same time. This makes threads useless for intense algorithmic work. The multiprocessing module bypasses the lock entirely by spawning separate operating system processes. Each process has its own Python interpreter and memory space, enabling true parallelism across all processing cores.

-   :material-cog-outline: __[Advanced Python - OOP](Advanced Python - OOP.md)__

    Object-Oriented Programming (OOP) is essential for building scalable, maintainable trading systems and financial applications. Learn to organize code using classes, objects, and OOP principles.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
