"""
Monte Carlo Portfolio Simulator
-------------------------------
This tool uses random simulation to show how a portfolio might grow or shrink over time—one of the core ideas in modern finance!
"""

import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_sim(initial_investment, mu, sigma, periods, simulations=1000):
    """
    Run Monte Carlo simulations for a portfolio.
    Args:
        initial_investment (float): Starting money
        mu (float): Expected average return per period (decimal, e.g., 0.001 for 0.1% per day)
        sigma (float): Standard deviation per period (volatility)
        periods (int): Time steps in each simulation (e.g. 252 trading days)
        simulations (int): Number of simulations to run
    Returns:
        np.array: Final simulated portfolio values
    """
    results = np.zeros(simulations)
    for i in range(simulations):
        rand_returns = np.random.normal(mu, sigma, periods)
        # Simulate growth with compounding
        series = initial_investment * np.cumprod(1 + rand_returns)
        results[i] = series[-1]
    return results


def plot_monte_carlo(
    initial_investment,
    mu,
    sigma,
    periods,
    simulations=100,
    title="Portfolio Growth Scenarios",
):
    plt.figure(figsize=(10, 6))
    for _ in range(simulations):
        returns = np.random.normal(mu, sigma, periods)
        series = initial_investment * np.cumprod(1 + returns)
        plt.plot(series, color="skyblue", alpha=0.3)
    plt.title(title)
    plt.xlabel("Time Period")
    plt.ylabel("Portfolio Value")
    plt.show()


if __name__ == "__main__":
    # Example usage: Simulate 10 years, 252 days/year, 1000 scenarios
    init = 10000
    yearly_mu = 0.07  # 7% mean return
    yearly_sigma = 0.15  # 15% vol
    periods = 10 * 252
    daily_mu = yearly_mu / 252
    daily_sigma = yearly_sigma / np.sqrt(252)

    # Histogram of outcomes (terminal wealth)
    results = monte_carlo_sim(init, daily_mu, daily_sigma, periods)
    import matplotlib.pyplot as plt

    plt.hist(results, bins=50, color="lightgreen")
    plt.title("Possible Portfolio Values in 10 Years")
    plt.xlabel("Ending Value")
    plt.ylabel("Frequency")
    plt.show()
    # Plot example paths
    plot_monte_carlo(init, daily_mu, daily_sigma, periods, simulations=100)
