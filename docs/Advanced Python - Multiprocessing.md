# Advanced Python Multiprocessing

## Overview

Python Global Interpreter Lock prevents multiple threads from executing Python bytecode at the same time. This makes threads useless for intense algorithmic work. The multiprocessing module bypasses the lock entirely by spawning separate operating system processes. Each process has its own Python interpreter and memory space, enabling true parallelism across all processing cores.

In quantitative finance, multiprocessing dramatically reduces computation time for:
*   Monte Carlo simulations
*   Strategy backtests across many tickers or time periods
*   Parameter grid searches and optimisation

## System Architecture Diagram

This drawing indicates how tasks are distributed across individual processing units to resolve computational bottlenecks.

```text
                   [ Main Python Engine ]
                      (Coordinates Work)
                             |
          +------------------+------------------+
          |                  |                  |
   [ Worker Core 1 ]  [ Worker Core 2 ]  [ Worker Core 3 ]
   (Isolated Memory)  (Isolated Memory)  (Isolated Memory)
          |                  |                  |
    Path Sim 1          Path Sim 2         Path Sim 3
    Path Sim 4          Path Sim 5         Path Sim 6
          |                  |                  |
          +------------------+------------------+
                             |
                   [ Aggregated Results ]
```

## Key Concepts

### Global Interpreter Lock Importance

*   Intense math computations require multiprocessing to bypass the lock
*   File reading or networking require threading because the lock is released during waits

### ProcessPoolExecutor

High level interface:
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(my_function, args_list))
```

### Completely Parallel Problems

Tasks with zero shared dependencies are perfect candidates for parallelism:
*   Each Monte Carlo path is independent
*   Each ticker backtest is independent
*   Each parameter combination is independent

### Pickling Requirement

Everything passed to a worker process must be picklable:
*   Functions must be defined at module level
*   Arguments should be simple types
*   Use unique seeds per worker for statistical independence

## Logic Implemented

1. Single path returning terminal price
2. Baseline run passing all simulations on one core
3. Processing pool distributing work across cores
4. Array based Monte Carlo option pricer
5. Grid search across volatility levels

## Files

`multiprocessing_tutorial.py` simulates price paths and compares sequential execution versus parallel execution.

## How to Run

```bash
python multiprocessing_tutorial.py
```

Important Note: Always guard module level code against accidental triggering upon import.

## Financial Applications

### Monte Carlo Risk Simulation

*   Value at Risk via thousands of portfolio simulations
*   Credit risk models simulating correlated default scenarios
*   American option pricing via complex path dependent algorithms

### Strategy Backtesting at Scale

*   Backtest a strategy on five hundred stocks simultaneously
*   Walk forward optimisation across overlapping time windows
*   Regime conditional backtests for multiple market environments

### Model Calibration

*   Calibrate stochastic volatility models to implied volatility surfaces
*   Each candidate parameter set is evaluated independently

### Machine Learning Feature Generation

*   Computing technical indicators for thousands of tickers
*   Generating historical feature matrices in parallel

## Best Practices

*   Task granularity: Each task should take meaningful time to outweigh process initialization overhead.
*   Data passing: Prefer passing small arguments rather than large data frames to minimize transmission overhead.
*   Error handling: Capture futures carefully to reraise exceptions from worker processes.
*   Processing count: Set workers to the maximum parallelism on the local machine.
*   Memory: Each process gets a copy of all data, so watch for memory explosion.