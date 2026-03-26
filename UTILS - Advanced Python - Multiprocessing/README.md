# Advanced Python – Multiprocessing

## Overview

Python's **Global Interpreter Lock (GIL)** prevents multiple threads from executing Python bytecode at the same time, making threads useless for CPU-bound work. The `multiprocessing` module bypasses the GIL entirely by spawning *separate OS processes*, each with its own Python interpreter and memory space — enabling true parallelism across all CPU cores.

In quantitative finance, multiprocessing dramatically reduces computation time for:
- Monte Carlo simulations (thousands of independent paths)
- Strategy backtests across many tickers or time periods
- Parameter grid searches and optimisation

## Key Concepts

### **GIL and Why It Matters**
| Task Type | Use This | Why |
|-----------|----------|-----|
| CPU-bound (simulations, math) | `multiprocessing` | Bypasses the GIL |
| I/O-bound (API calls, file reads) | `threading` or `asyncio` | GIL released during I/O waits |

### **ProcessPoolExecutor**
High-level API from `concurrent.futures`:
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
 results = list(executor.map(my_function, args_list))
```

### **Embarrassingly Parallel Problems**
Tasks with **zero inter-dependencies** — perfect candidates for parallelism:
- Each Monte Carlo path is independent → ideal for process pools
- Each ticker backtest is independent → ideal for process pools
- Each parameter combination is independent → ideal for parameter sweeps

### **Pickling Requirement**
Everything passed to a worker process must be *picklable*:
- Functions must be defined at module level (no lambdas)
- Arguments should be simple types (tuples, arrays, dicts)
- Use unique seeds per worker for statistical independence

## Logic Implemented

1. **`simulate_gbm_path(args)`** — Single GBM path returning terminal price
2. **`run_sequential()`** — Baseline: all simulations on one core
3. **`run_parallel()`** — ProcessPoolExecutor distributing work across cores
4. **`price_call_option_mc()`** — Vectorised Monte Carlo option pricer per volatility
5. **`parallel_parameter_sweep()`** — Grid search across vol levels using `as_completed()`

## Files
- `multiprocessing_tutorial.py`: GBM simulation, sequential vs. parallel comparison, and option pricing parameter sweep.

## How to Run
```bash
python multiprocessing_tutorial.py
```

> **Windows Note:** Always guard module-level code with `if __name__ == "__main__":`. Windows uses *spawn* (not *fork*) to create processes, so the module is re-imported in each worker — any top-level side-effects would run in every process.

## Financial Applications

### 1. Monte Carlo Risk Simulation
- VaR and CVaR via thousands of portfolio simulations
- Credit risk models simulating correlated default scenarios
- American option pricing via Longstaff-Schwartz (path-dependent, computationally heavy)

### 2. Strategy Backtesting at Scale
- Backtest a strategy on 500 US stocks simultaneously
- Walk-forward optimisation across overlapping time windows
- Regime-conditional backtests for multiple market environments

### 3. Model Calibration
- Calibrate Heston, SABR, or other stochastic vol models to implied vol surfaces
- Each candidate parameter set is evaluated independently → natural process pool

### 4. Machine Learning Feature Generation
- Computing technical indicators for thousands of tickers
- Generating lagged feature matrices in parallel

## Best Practices

- **Task granularity**: Each task should take ≥100ms; process spawn overhead (~50–200ms) dominates for tiny tasks.
- **Data passing**: Prefer passing small arguments (scalars, seeds) rather than large DataFrames to minimise serialisation overhead.
- **Error handling**: Use `future.result()` to re-raise exceptions from worker processes.
- **CPU count**: Set `max_workers=os.cpu_count()` for maximum parallelism on the local machine.
- **Memory**: Each process gets a *copy* of all data — watch for memory bloat with large shared arrays (use `multiprocessing.shared_memory` for advanced cases).