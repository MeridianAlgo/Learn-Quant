<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Advanced Python</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Advanced Python - Decorators and Generators"
    python "decorators_generators_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Advanced%20Python%20-%20Decorators%20and%20Generators)

---
# Advanced Python – Decorators and Generators

## Overview

Decorators and Generators are powerful Python features that separate professional code from beginner scripts. Decorators allow you to modify function behavior cleanly, while Generators enable memory-efficient processing of large financial datasets.

## Key Concepts

### **Decorators `@wrapper`**
- **Function Wrappers**: Modify input/output without changing code
- **Cross-Cutting Concerns**: Logging, timing, authentication, error handling
- **Syntax Sugar**: `@decorator` is equivalent to `func = decorator(func)`
- **`functools.wraps`**: Preserves original function metadata

### **Generators `yield`**
- **Lazy Evaluation**: Compute values only when needed
- **Memory Efficient**: Process terabytes of data with minimal RAM
- **Infinite Streams**: Model real-time data feeds
- **Pipelines**: Chain generators for modular data processing

## Key Examples

### Timing Decorator
```python
def timer(func):
 @functools.wraps(func)
 def wrapper(*args, **kwargs):
 start = time.time()
 result = func(*args, **kwargs)
 print(f"Took {time.time() - start:.4f}s")
 return result
 return wrapper

@timer
def heavy_calc():
 # ...
```

### Simple Generator
```python
def price_stream():
 price = 100
 while True:
 price += random.uniform(-1, 1)
 yield price

# Usage
stream = price_stream()
print(next(stream)) # 100.5
print(next(stream)) # 99.8
```

### Generator Pipeline
```python
raw_data = read_csv_generator("trades.csv")
filtered = (t for t in raw_data if t['symbol'] == 'AAPL')
processed = (process_trade(t) for t in filtered)

for trade in processed:
 save_to_db(trade)
```

## Files
- `decorators_generators_tutorial.py`: Interactive tutorial

## How to Run
```bash
python decorators_generators_tutorial.py
```

## Financial Applications

### 1. Robust API Calls (Retry Decorator)
Automatically retry failed API requests with exponential backoff:
```python
@retry(attempts=3, delay=1)
def get_market_data(ticker):
 # ...
```

### 2. Caching (Memoization)
Cache expensive calculations (like implied volatility) to speed up backtests:
```python
@lru_cache(maxsize=1000)
def black_scholes(S, K, T, r, sigma):
 # ...
```

### 3. Streaming Backtest (Generators)
Process tick data year-by-year without loading everything into RAM:
```python
def tick_generator(file_path):
 with open(file_path) as f:
 for line in f:
 yield parse_tick(line)
```

### 4. Event-Driven Systems
Use coroutines (generators that accept input) to model strategy logic:
```python
def strategy():
 while True:
 market_data = yield
 if market_data.price > 100:
 yield "BUY"
```

## Best Practices

- **Use `yield from`**: Delegate to sub-generators.
- **Avoid Side Effects**: Decorators should generally be transparent.
- **Generator Expressions**: Use `(x for x in data)` instead of `[x for x in data]` for large sequences.
- **Debugging**: Decorators can make stack traces harder to read; use `functools.wraps`.

---

*Master these advanced features to write professional, scalable, and efficient financial software!*

---

## Continue in Advanced Python

<div class="grid cards" markdown>

-   :material-cog-outline: __[Advanced Python - AsyncIO](Advanced Python - AsyncIO.md)__

    In quantitative finance, speed is edge. Python's `asyncio` library allows for **concurrency**, letting your program handle multiple tasks (like fetching data from 10 different exchanges) at once, rather than waiting for one to finish before starting the next.

-   :material-cog-outline: __[Advanced Python - Context Managers](Advanced Python - Context Managers.md)__

    Context Managers are a powerful Python feature for resource management. They allow you to allocate and release resources precisely when you want to. The most common usage is the `with` statement.

-   :material-cog-outline: __[Advanced Python - Error Handling](Advanced Python - Error Handling.md)__

    Robust error handling is what separates a script that crashes overnight from a professional trading system that runs for years. This module teaches you how to anticipate, catch, and manage errors gracefully.

-   :material-cog-outline: __[Advanced Python - Multiprocessing](Advanced Python - Multiprocessing.md)__

    Python Global Interpreter Lock prevents multiple threads from executing Python bytecode at the same time. This makes threads useless for intense algorithmic work. The multiprocessing module bypasses the lock entirely by spawning separate operating system processes. Each process has its own Python interpreter and memory space, enabling true parallelism across all processing cores.

-   :material-cog-outline: __[Advanced Python - OOP](Advanced Python - OOP.md)__

    Object-Oriented Programming (OOP) is essential for building scalable, maintainable trading systems and financial applications. Learn to organize code using classes, objects, and OOP principles.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
