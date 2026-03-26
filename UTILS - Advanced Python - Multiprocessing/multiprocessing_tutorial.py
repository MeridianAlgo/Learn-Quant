"""Python Multiprocessing for Quantitative Finance.

Run with:
    python multiprocessing_tutorial.py

Python's Global Interpreter Lock (GIL) prevents multiple threads from running
Python bytecode simultaneously.  Threads don't speed up CPU-bound computation.
Multiprocessing spawns *separate OS processes*, each with its own interpreter
and memory space, bypassing the GIL entirely and using all available CPU cores.

In quantitative finance, multiprocessing is the standard tool for:
- Running thousands of Monte Carlo simulation paths in parallel.
- Backtesting a strategy across many tickers or parameter combinations simultaneously.
- Optimising strategy parameters by evaluating a large grid in parallel.

Key Concepts:
- Process vs Thread: Processes have independent memory; threads share memory.
- GIL: Python's Global Interpreter Lock limits CPU-bound thread parallelism.
- ProcessPoolExecutor: High-level concurrent.futures API for parallel work distribution.
- Embarrassingly Parallel: Problems with no inter-task dependencies (ideal for pools).
- Chunking: Splitting a large workload into equal pieces, one per worker process.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ─── 1. SINGLE GBM PATH ──────────────────────────────────────────────────────

def simulate_gbm_path(args: tuple) -> float:
    """
    Simulate one Geometric Brownian Motion (GBM) price path and return its terminal price.

    GBM is the standard model for stock price dynamics underlying Black-Scholes:
        dS = mu * S * dt + sigma * S * dW

    In discrete form for N steps of size dt = T/N:
        S(T) = S(0) * exp( sum of [ (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_i ] )

    This function takes a single tuple argument so it can be mapped cleanly
    via ProcessPoolExecutor.map() without lambda expressions (which can't be pickled).

    Args:
        args: Tuple of (seed, S0, mu, sigma, T, n_steps) where:
            seed    – integer seed for reproducibility (unique per simulation)
            S0      – initial stock price
            mu      – annual drift (expected return)
            sigma   – annual volatility
            T       – time horizon in years
            n_steps – number of discrete time steps (252 = daily over 1 year)

    Returns:
        Terminal stock price S(T) as a float.
    """
    seed, S0, mu, sigma, T, n_steps = args

    # numpy's new default_rng is thread/process-safe; each process gets a unique seed
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Each step: log return ~ N( (mu - 0.5*sigma^2)*dt,  sigma^2*dt )
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_steps)

    # Terminal price: S0 * exp( sum of all log returns )
    terminal_price = S0 * np.exp(np.sum(log_returns))
    return terminal_price


# ─── 2. SEQUENTIAL (SINGLE-CORE) SIMULATION ──────────────────────────────────

def run_sequential(n_simulations: int, S0: float, mu: float, sigma: float,
                   T: float, n_steps: int) -> list:
    """
    Run Monte Carlo GBM simulations one at a time on a single CPU core.

    This is the naive baseline — useful for correctness verification and timing
    comparisons.  All work happens on a single core, no parallelism.

    Args:
        n_simulations: Total number of price paths to simulate.
        S0, mu, sigma, T, n_steps: GBM parameters (see simulate_gbm_path).

    Returns:
        List of terminal prices (one float per simulation).
    """
    # Build the argument tuple for each simulation, using a unique seed per run
    args_list = [(i, S0, mu, sigma, T, n_steps) for i in range(n_simulations)]
    return [simulate_gbm_path(args) for args in args_list]


# ─── 3. PARALLEL (MULTI-CORE) SIMULATION ────────────────────────────────────

def run_parallel(n_simulations: int, S0: float, mu: float, sigma: float,
                 T: float, n_steps: int, max_workers: int = 4) -> list:
    """
    Run Monte Carlo GBM simulations in parallel across multiple CPU cores.

    ProcessPoolExecutor manages a pool of worker processes.  Work items from
    args_list are automatically distributed across workers so all cores stay busy.

    Design choices:
    - Each task is a self-contained tuple (no shared state between workers).
    - Unique seeds per simulation guarantee statistically independent paths.
    - The 'with' context manager ensures proper process pool cleanup on exit.

    Args:
        n_simulations: Total number of price paths to simulate.
        S0, mu, sigma, T, n_steps: GBM parameters.
        max_workers:   Number of worker processes (try os.cpu_count() for maximum).

    Returns:
        List of terminal prices in the same order as the input args.
    """
    args_list = [(i, S0, mu, sigma, T, n_steps) for i in range(n_simulations)]

    # 'with' ensures worker processes are terminated even if an exception occurs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map() distributes the args_list and collects results in order
        results = list(executor.map(simulate_gbm_path, args_list))

    return results


# ─── 4. OPTION PRICING WORKER ────────────────────────────────────────────────

def price_call_option_mc(args: tuple) -> dict:
    """
    Price a European call option via Monte Carlo for a single volatility value.

    European call payoff at expiry: max(S(T) – K, 0)
    Monte Carlo estimate:           exp(–r*T) * mean( max(S_i(T) – K, 0) )

    This is an "embarrassingly parallel" workload — each volatility level is
    completely independent and can be computed on a separate CPU core.

    Args:
        args: Tuple of (sigma, S0, K, r, T, n_sims, seed_offset).

    Returns:
        Dict with 'sigma' and 'call_price' keys.
    """
    sigma, S0, K, r, T, n_sims, seed_offset = args

    rng = np.random.default_rng(seed_offset)
    n_steps = 252   # daily steps over the option life
    dt = T / n_steps

    # Vectorised GBM: simulate all paths at once using a (n_sims × n_steps) array
    # Each row is one simulation path; columns are daily log returns
    Z = rng.standard_normal((n_sims, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Terminal prices: S0 * exp( sum of log returns for each path )
    terminal_prices = S0 * np.exp(np.sum(log_returns, axis=1))

    # Call option payoff under risk-neutral pricing (use r, not mu, for drift)
    payoffs = np.maximum(terminal_prices - K, 0.0)

    # Discount expected payoff back to today (present value)
    call_price = float(np.exp(-r * T) * np.mean(payoffs))

    return {"sigma": sigma, "call_price": call_price}


# ─── 5. PARALLEL PARAMETER SWEEP ────────────────────────────────────────────

def parallel_parameter_sweep(sigmas: list, S0: float, K: float, r: float,
                              T: float, n_sims: int = 5000,
                              max_workers: int = 4) -> pd.DataFrame:
    """
    Price a European call option across a range of volatilities in parallel.

    Parameter sweeps (grid searches, scenario analyses, stress tests) are among
    the most common use cases for multiprocessing in quant finance.  Here we
    price the same option at many different implied volatility levels simultaneously.

    Strategy:
    - Submit all tasks to the pool with executor.submit() (non-blocking).
    - Collect results with as_completed() which yields futures as they finish.
    - Re-sort by sigma afterward because as_completed() returns in arrival order.

    Args:
        sigmas:      List of annual volatility values to sweep (e.g., [0.10, 0.20, 0.30]).
        S0:          Initial stock price.
        K:           Strike price of the call option.
        r:           Annual risk-free interest rate.
        T:           Time to expiry in years.
        n_sims:      Number of Monte Carlo paths per volatility level.
        max_workers: Number of parallel worker processes.

    Returns:
        DataFrame sorted by sigma with columns: sigma, call_price.
    """
    # Build args: each tuple contains everything a worker needs — no shared state
    args_list = [(s, S0, K, r, T, n_sims, i * 1000) for i, s in enumerate(sigmas)]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs at once (non-blocking)
        futures = [executor.submit(price_call_option_mc, args) for args in args_list]

        # Collect results as each finishes (order not guaranteed)
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by sigma to restore natural order after as_completed() reorders results
    results.sort(key=lambda x: x["sigma"])
    return pd.DataFrame(results)


# ─── 6. TIMING UTILITY ───────────────────────────────────────────────────────

def timed(label: str, func, *args, **kwargs):
    """Run func(*args, **kwargs), print elapsed time, and return the result."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {label:<25} {elapsed:.3f}s")
    return result


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("MULTIPROCESSING FOR QUANTITATIVE FINANCE")
    print("=" * 60)

    # GBM parameters used throughout this tutorial
    S0 = 100.0      # starting stock price
    mu = 0.08       # 8% annual expected return (drift)
    sigma = 0.20    # 20% annual volatility
    T = 1.0         # 1-year simulation horizon
    n_steps = 252   # one step per trading day
    n_sims = 2000   # number of Monte Carlo paths (kept small for tutorial speed)

    # ── Demo 1: Sequential vs. parallel speed comparison ──
    print(f"\n[1] Speed Comparison: {n_sims} GBM simulations, {n_steps} steps each")
    print("  Timing (wall clock):")

    seq_prices = timed("Sequential (1 core):", run_sequential,
                       n_sims, S0, mu, sigma, T, n_steps)

    par_prices = timed("Parallel  (4 cores):", run_parallel,
                       n_sims, S0, mu, sigma, T, n_steps, max_workers=4)

    # Both approaches should give statistically similar means (same seeds)
    print(f"\n  Sequential mean terminal price:  ${np.mean(seq_prices):.2f}")
    print(f"  Parallel   mean terminal price:  ${np.mean(par_prices):.2f}")
    print("  (Means differ slightly because parallelism changes execution order but not seeds)")

    # ── Demo 2: Monte Carlo option pricing sweep ──
    print("\n[2] Parallel Parameter Sweep: European Call Price vs. Volatility")
    print("  Pricing a call option (S0=100, K=100, r=5%, T=1yr) at 7 vol levels:")

    sigmas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sweep_df = parallel_parameter_sweep(
        sigmas, S0=100.0, K=100.0, r=0.05, T=1.0, n_sims=4000, max_workers=4
    )
    print(sweep_df.to_string(index=False, float_format="{:.4f}".format))
    print("  Observation: Higher volatility → higher call price (classic Black-Scholes result).")

    # ── Key takeaways ──
    print("\n[3] When to Use What:")
    print("  Multiprocessing  → CPU-bound: simulations, backtests, model fitting")
    print("  Multithreading   → I/O-bound: REST API calls, file reads, database queries")
    print("  Async/Await      → I/O-bound: WebSocket streams, high-concurrency I/O")
    print()
    print("  ProcessPoolExecutor tips:")
    print("  - Keep tasks large: spawning a process has ~50-200ms overhead on Windows.")
    print("  - Avoid shared mutable state: each process gets a copy of data passed to it.")
    print("  - Use unique seeds per worker to ensure statistical independence.")
    print("  - On Windows, always guard top-level code with if __name__ == '__main__'.")

    print("\nMultiprocessing tutorial complete!")


if __name__ == "__main__":
    main()
