# Advanced Python – Decorators and Generators

## 📋 Overview

Decorators and Generators are powerful Python features that separate professional code from beginner scripts. Decorators allow you to modify function behavior cleanly, while Generators enable memory-efficient processing of large financial datasets.

## 🎯 Key Concepts

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

## 💻 Key Examples

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
print(next(stream))  # 100.5
print(next(stream))  # 99.8
```

### Generator Pipeline
```python
raw_data = read_csv_generator("trades.csv")
filtered = (t for t in raw_data if t['symbol'] == 'AAPL')
processed = (process_trade(t) for t in filtered)

for trade in processed:
    save_to_db(trade)
```

## 📂 Files
- `decorators_generators_tutorial.py`: Interactive tutorial

## 🚀 How to Run
```bash
python decorators_generators_tutorial.py
```

## 🧠 Financial Applications

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

## 💡 Best Practices

- **Use `yield from`**: Delegate to sub-generators.
- **Avoid Side Effects**: Decorators should generally be transparent.
- **Generator Expressions**: Use `(x for x in data)` instead of `[x for x in data]` for large sequences.
- **Debugging**: Decorators can make stack traces harder to read; use `functools.wraps`.

---

*Master these advanced features to write professional, scalable, and efficient financial software!*
