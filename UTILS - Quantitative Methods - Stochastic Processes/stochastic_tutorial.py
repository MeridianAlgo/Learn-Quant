"""Stochastic Processes Tutorial for Quantitative Finance.

Run with:
    python stochastic_tutorial.py

This module teaches Brownian Motion, Geometric Brownian Motion (GBM),
and Mean Reversion (Ornstein-Uhlenbeck) processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("QUANTITATIVE METHODS – STOCHASTIC PROCESSES")
    print("#" * 60)
    print("Simulating random asset price paths using:")
    print("1. Brownian Motion (Random Walk)")
    print("2. Geometric Brownian Motion (Stock Prices)")
    print("3. Ornstein-Uhlenbeck (Mean Reversion)\n")


def brownian_motion() -> None:
    """Simulate standard Brownian Motion (Wiener Process)."""
    print("=" * 60)
    print("BROWNIAN MOTION (Wiener Process)")
    print("=" * 60)
    
    # Parameters
    T = 1.0         # Time horizon (1 year)
    N = 252         # Number of steps (daily)
    dt = T / N      # Time step
    n_sims = 5      # Number of simulations
    
    print(f"Time Horizon: {T} years")
    print(f"Steps: {N}")
    print(f"Simulations: {n_sims}")
    
    # Generate random increments: dW ~ N(0, dt)
    # Scale by sqrt(dt) because variance scales linearly with time
    np.random.seed(42)
    dW = np.random.normal(0, np.sqrt(dt), (n_sims, N))
    
    # Cumulative sum to get Brownian path W(t)
    # W(0) = 0
    W = np.cumsum(dW, axis=1)
    # Add starting zero
    W = np.hstack([np.zeros((n_sims, 1)), W])
    
    print("\nFinal Values W(T):")
    for i in range(n_sims):
        print(f"  Sim {i+1}: {W[i, -1]:.4f}")
    
    print(f"\nExpected Mean: 0.00")
    print(f"Actual Mean: {np.mean(W[:, -1]):.4f}")
    print(f"Expected Std Dev (sqrt(T)): {np.sqrt(T):.4f}")
    print(f"Actual Std Dev: {np.std(W[:, -1]):.4f}")


def geometric_brownian_motion() -> None:
    """Simulate Geometric Brownian Motion (Stock Prices)."""
    print("\n" + "=" * 60)
    print("GEOMETRIC BROWNIAN MOTION (GBM)")
    print("=" * 60)
    
    # dS = μSdt + σSdW
    # S(t) = S(0) * exp((μ - 0.5σ²)t + σW(t))
    
    # Parameters
    S0 = 100.0      # Initial price
    mu = 0.08       # Drift (Expected Return)
    sigma = 0.20    # Volatility
    T = 1.0         # Time horizon
    N = 252         # Steps
    dt = T / N
    n_sims = 1000   # Many sims for statistics
    
    print(f"Initial Price: ${S0}")
    print(f"Drift (μ): {mu:.1%}")
    print(f"Volatility (σ): {sigma:.1%}")
    
    # Simulation
    np.random.seed(42)
    
    # Generate random component
    # Z ~ N(0, 1)
    Z = np.random.normal(0, 1, (n_sims, N))
    
    # Calculate price paths
    # S_t = S_{t-1} * exp((μ - 0.5σ²)dt + σ*sqrt(dt)*Z)
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * Z
    
    # Daily returns (log returns)
    daily_log_returns = drift_term + diffusion_term
    
    # Accumulate returns
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=1)
    
    # Calculate prices
    S = S0 * np.exp(cumulative_log_returns)
    S = np.hstack([np.full((n_sims, 1), S0), S])
    
    # Statistics
    final_prices = S[:, -1]
    
    print("\nSimulation Results (1000 paths):")
    print(f"  Mean Final Price: ${np.mean(final_prices):.2f}")
    print(f"  Theoretical Mean ($100 * e^0.08): ${S0 * np.exp(mu * T):.2f}")
    
    print(f"  Min Price: ${np.min(final_prices):.2f}")
    print(f"  Max Price: ${np.max(final_prices):.2f}")
    
    # Probability of profit
    prob_profit = np.mean(final_prices > S0)
    print(f"  Probability of Profit: {prob_profit:.1%}")


def mean_reversion_ou() -> None:
    """Simulate Ornstein-Uhlenbeck Process (Mean Reversion)."""
    print("\n" + "=" * 60)
    print("ORNSTEIN-UHLENBECK (Mean Reversion)")
    print("=" * 60)
    
    # dx = θ(μ - x)dt + σdW
    # Used for volatility, interest rates, pairs trading spread
    
    # Parameters
    theta = 2.0     # Speed of reversion
    mu = 0.0        # Long-term mean
    sigma = 0.3     # Volatility
    x0 = 2.0        # Initial value (far from mean)
    T = 2.0         # Time horizon
    N = 500         # Steps
    dt = T / N
    
    print(f"Speed of Reversion (θ): {theta}")
    print(f"Long-term Mean (μ): {mu}")
    print(f"Initial Value: {x0}")
    
    # Simulation
    np.random.seed(42)
    x = np.zeros(N + 1)
    x[0] = x0
    
    for t in range(1, N + 1):
        # Euler-Maruyama discretization
        dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x[t] = x[t-1] + dx
    
    print("\nPath Analysis:")
    print(f"  Start: {x[0]:.4f}")
    print(f"  End: {x[-1]:.4f}")
    print(f"  Average: {np.mean(x):.4f} (should be close to {mu})")
    
    # Check how often it crosses the mean
    crossings = 0
    for i in range(1, len(x)):
        if (x[i-1] > mu and x[i] < mu) or (x[i-1] < mu and x[i] > mu):
            crossings += 1
            
    print(f"  Mean Crossings: {crossings}")


def jump_diffusion() -> None:
    """Simulate Merton Jump Diffusion Model."""
    print("\n" + "=" * 60)
    print("MERTON JUMP DIFFUSION")
    print("=" * 60)
    
    # GBM + Poisson Jumps
    # Used for crypto, earnings announcements, crashes
    
    # Parameters
    S0 = 100.0
    mu = 0.05
    sigma = 0.20
    lambda_jump = 2.0    # Avg jumps per year
    mu_jump = -0.10      # Avg jump size (-10%)
    sigma_jump = 0.05    # Jump volatility
    T = 1.0
    N = 252
    dt = T / N
    
    print(f"Jump Intensity (λ): {lambda_jump} per year")
    print(f"Avg Jump Size: {mu_jump:.1%}")
    
    np.random.seed(42)
    
    # Standard GBM component
    Z = np.random.normal(0, 1, N)
    gbm_log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    # Jump component
    # Poisson process for number of jumps
    # Note: For small dt, prob of jump ≈ λ*dt
    jumps = np.random.poisson(lambda_jump * dt, N)
    
    jump_log_ret = np.zeros(N)
    jump_count = 0
    
    for i in range(N):
        if jumps[i] > 0:
            # Simulate jump size
            jump_size = np.random.normal(mu_jump, sigma_jump, jumps[i])
            jump_log_ret[i] = np.sum(jump_size)
            jump_count += jumps[i]
    
    # Total returns
    total_log_ret = gbm_log_ret + jump_log_ret
    path = S0 * np.exp(np.cumsum(total_log_ret))
    path = np.insert(path, 0, S0)
    
    print(f"\nTotal Jumps: {jump_count}")
    print(f"Final Price: ${path[-1]:.2f}")
    print(f"Min Price: ${np.min(path):.2f}")


def main() -> None:
    """Run all stochastic process examples."""
    intro()
    brownian_motion()
    geometric_brownian_motion()
    mean_reversion_ou()
    jump_diffusion()
    print("\n🎉 Stochastic Processes tutorial complete!")
    print("These models form the basis of derivatives pricing and risk management.")


if __name__ == "__main__":
    main()
