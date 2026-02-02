# Advanced Python – Context Managers

## 📋 Overview

Context Managers are a powerful Python feature for resource management. They allow you to allocate and release resources precisely when you want to. The most common usage is the `with` statement.

In financial applications, they are essential for:
- Ensuring database connections are closed.
- Handling atomic transactions (commit/rollback).
- Timing execution of strategy code.
- Managing thread locks for thread-safe trading bots.

## 🎯 Key Concepts

### **The `with` Statement**
- Simplifies exception handling by encapsulating standard uses of `try...finally`.
- Ensures clean-up code is executed automatically.

### **Class-Based Context Managers**
- Implement `__enter__`: Setup code, returns the object used in `as`.
- Implement `__exit__`: Teardown code, handles exceptions.

### **Generator-Based Context Managers**
- Use `@contextlib.contextmanager` decorator.
- Write a generator with a single `yield`.
- Code before `yield` is setup; code after is teardown.

## 💻 Key Examples

### Class-Based Example
```python
class ManagedFile:
    def __init__(self, filename):
        self.filename = filename
        
    def __enter__(self):
        self.file = open(self.filename, 'w')
        return self.file
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

with ManagedFile('log.txt') as f:
    f.write('Trade executed.')
```

### Function-Based Example
```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("Acquiring resource...")
    yield resource
    print("Releasing resource...")

with managed_resource() as r:
    use(r)
```

## 📂 Files
- `context_managers_tutorial.py`: Tutorial script demonstrating timers, custom generators, and transaction rollbacks.

## 🚀 How to Run
```bash
python context_managers_tutorial.py
```

## 🧠 Financial Applications

### 1. Atomic Portfolio Updates
Ensure that a sequence of portfolio changes either all succeed or all fail (rollback).
```python
with Transaction(portfolio) as p:
    p.deduct_cash(1000)
    p.add_stock('AAPL', 10)
# If error occurs in add_stock, cash is refunded automatically.
```

### 2. High-Frequency execution timing
Measure exactly how long a signal generation step takes.
```python
with Timer("SignalGeneration"):
    strategy.calculate_signals()
```

### 3. Database Sessions
Automatically close connections to market data databases.
```python
with DBConnection("market_data.db") as conn:
    data = conn.query("SELECT * FROM prices")
```

## 💡 Best Practices

- **Use `contextlib.suppress`**: To explicitly ignore specific errors.
- **Return `True` in `__exit__`**: Only if you intend to suppress the exception.
- **Keep it simple**: For simple setup/teardown, use `@contextmanager`. For complex state, use a Class.
