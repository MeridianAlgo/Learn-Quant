# -----------------------------
# Options Chain Simulator Utility (NO API)
# -----------------------------
# This script lets you simulate options pricing and Greeks for learning.
# All calculations are done locally (no APIs, no real market data).
# All code is commented for beginners to learn Python and finance.
# -----------------------------

import math

# Helper function: Cumulative distribution function for standard normal
# Used in Black-Scholes formula

def norm_cdf(x):
    """Cumulative distribution function for standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# Black-Scholes pricing formula for European options
# S: stock price, K: strike, T: time to expiry (years), r: risk-free rate, sigma: volatility
# option_type: 'call' or 'put'
def black_scholes_price(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return price, d1, d2

# Calculate Greeks for European options
# Returns a dictionary with price, delta, gamma, theta, vega, rho
def black_scholes_greeks(S, K, T, r, sigma, option_type):
    price, d1, d2 = black_scholes_price(S, K, T, r, sigma, option_type)
    nd1 = math.exp(-d1**2/2) / math.sqrt(2*math.pi)
    if option_type == 'call':
        delta = norm_cdf(d1)
        theta = (-S * nd1 * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
        rho = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
    else:
        delta = -norm_cdf(-d1)
        theta = (-S * nd1 * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100
    gamma = nd1 / (S * sigma * math.sqrt(T))
    vega = S * nd1 * math.sqrt(T) / 100
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Calculate the payoff for a call or put option at a given stock price
# Used for text-based payoff diagrams
def payoff(option_type, S, K):
    if option_type == 'call':
        return max(S - K, 0)
    else:
        return max(K - S, 0)

# Print the main menu for the CLI
def print_menu():
    print("\nOptions Chain Simulator Menu:")
    print("1. Set parameters (stock price, volatility, rate, expiry)")
    print("2. Generate options chain")
    print("3. View payoff diagram")
    print("4. Calculate Greeks for an option")
    print("5. Exit")

# Helper to get a float from user input
def input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main workflow for the CLI
def main():
    print("""
====================================
Welcome to the Options Chain Simulator!
This tool helps you learn Python and options math by simulating option prices and Greeks.
- No APIs or real market data are used.
- All code is commented for beginners.
====================================
""")
    S = 100.0  # Stock price
    sigma = 0.2  # Volatility
    r = 0.01  # Risk-free rate
    T = 0.5  # Expiry in years
    strikes = [80, 90, 100, 110, 120]
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            S = input_float("Enter stock price: ")
            sigma = input_float("Enter volatility (e.g. 0.2 for 20%): ")
            r = input_float("Enter risk-free rate (e.g. 0.05 for 5%): ")
            T = input_float("Enter expiry in years (e.g. 0.5): ")
            # Suggest some strikes around the stock price
            strikes = [round(S * x) for x in [0.8, 0.9, 1.0, 1.1, 1.2]]
            print(f"Available strikes: {', '.join(str(s) for s in strikes)}")
        elif choice == '2':
            print(f"\nOptions Chain (S={S}, sigma={sigma}, r={r}, T={T}):")
            print(f"{'Strike':>8} {'Call Price':>12} {'Put Price':>12}")
            for K in strikes:
                call = black_scholes_greeks(S, K, T, r, sigma, 'call')['price']
                put = black_scholes_greeks(S, K, T, r, sigma, 'put')['price']
                print(f"{K:>8} {call:>12.2f} {put:>12.2f}")
        elif choice == '3':
            K = input_float("Enter strike price: ")
            option_type = input("Option type (call/put): ").strip().lower()
            print("\nPayoff diagram (text, S from 0.5K to 1.5K):")
            for s in range(int(0.5*K), int(1.5*K)+1, int(K/10)):
                p = payoff(option_type, s, K)
                bar = '*' * int(p/2)
                print(f"S={s:>4} | Payoff: {p:>6.2f} {bar}")
        elif choice == '4':
            K = input_float("Enter strike price: ")
            option_type = input("Option type (call/put): ").strip().lower()
            greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
            print(f"\nOption Greeks for {option_type} (K={K}):")
            for k, v in greeks.items():
                print(f"{k.capitalize()}: {v:.4f}")
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
