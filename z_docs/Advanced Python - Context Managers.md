<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Advanced Python</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Advanced Python - Context Managers"
    python "context_managers_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Advanced%20Python%20-%20Context%20Managers)

---
# Advanced Python – Context Managers

## Overview

Context Managers are a powerful Python feature for resource management. They allow you to allocate and release resources precisely when you want to. The most common usage is the `with` statement.

In financial applications, they are essential for:
- Ensuring database connections are closed.
- Handling atomic transactions (commit/rollback).
- Timing execution of strategy code.
- Managing thread locks for thread-safe trading bots.

## Key Concepts

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

## Key Examples

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

## Files
- `context_managers_tutorial.py`: Tutorial script demonstrating timers, custom generators, and transaction rollbacks.

## How to Run
```bash
python context_managers_tutorial.py
```

## Financial Applications

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

## Best Practices

- **Use `contextlib.suppress`**: To explicitly ignore specific errors.
- **Return `True` in `__exit__`**: Only if you intend to suppress the exception.
- **Keep it simple**: For simple setup/teardown, use `@contextmanager`. For complex state, use a Class.

---

## Continue in Advanced Python

<div class="grid cards" markdown>

-   :material-cog-outline: __[Advanced Python - AsyncIO](Advanced Python - AsyncIO.md)__

    In quantitative finance, speed is edge. Python's `asyncio` library allows for **concurrency**, letting your program handle multiple tasks (like fetching data from 10 different exchanges) at once, rather than waiting for one to finish before starting the next.

-   :material-cog-outline: __[Advanced Python - Decorators and Generators](Advanced Python - Decorators and Generators.md)__

    Decorators and Generators are powerful Python features that separate professional code from beginner scripts. Decorators allow you to modify function behavior cleanly, while Generators enable memory-efficient processing of large financial datasets.

-   :material-cog-outline: __[Advanced Python - Error Handling](Advanced Python - Error Handling.md)__

    Robust error handling is what separates a script that crashes overnight from a professional trading system that runs for years. This module teaches you how to anticipate, catch, and manage errors gracefully.

-   :material-cog-outline: __[Advanced Python - Multiprocessing](Advanced Python - Multiprocessing.md)__

    Python Global Interpreter Lock prevents multiple threads from executing Python bytecode at the same time. This makes threads useless for intense algorithmic work. The multiprocessing module bypasses the lock entirely by spawning separate operating system processes. Each process has its own Python interpreter and memory space, enabling true parallelism across all processing cores.

-   :material-cog-outline: __[Advanced Python - OOP](Advanced Python - OOP.md)__

    Object-Oriented Programming (OOP) is essential for building scalable, maintainable trading systems and financial applications. Learn to organize code using classes, objects, and OOP principles.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
