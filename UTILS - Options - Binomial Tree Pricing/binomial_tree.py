import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

def binomial_tree_pricing(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, N=3, option_type='call', is_american=False):
    """
    Price an option using the Cox-Ross-Rubinstein Binomial Tree model.
    
    S: Initial stock price
    K: Strike price
    T: Time to maturity in years
    r: Risk-free rate
    sigma: Volatility
    N: Number of time steps
    option_type: 'call' or 'put'
    is_american: True for American options, False for European
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    ST = np.zeros(N + 1)
    for j in range(N + 1):
        ST[j] = S * (u ** j) * (d ** (N - j))
        
    # Initialize option values at maturity
    VT = np.zeros(N + 1)
    for j in range(N + 1):
        if option_type == 'call':
            VT[j] = max(0, ST[j] - K)
        else:
            VT[j] = max(0, K - ST[j])
            
    # Step backwards through the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Calculate option value based on risk-neutral expectation
            VT[j] = np.exp(-r * dt) * (q * VT[j + 1] + (1 - q) * VT[j])
            
            # Check for early exercise if American
            if is_american:
                ST_i = S * (u ** j) * (d ** (i - j))
                intrinsic_value = max(0, ST_i - K) if option_type == 'call' else max(0, K - ST_i)
                VT[j] = max(VT[j], intrinsic_value)
                
    return VT[0]

def main():
    console.print("[bold cyan]Binomial Tree Option Pricing[/bold cyan]")
    console.print("Pricing European and American Call/Put options using the Cox-Ross-Rubinstein model.\n")
    
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    N = 50 # 50 steps
    
    eur_call = binomial_tree_pricing(S, K, T, r, sigma, N, 'call', False)
    am_call = binomial_tree_pricing(S, K, T, r, sigma, N, 'call', True)
    eur_put = binomial_tree_pricing(S, K, T, r, sigma, N, 'put', False)
    am_put = binomial_tree_pricing(S, K, T, r, sigma, N, 'put', True)
    
    console.print(f"[green]European Call Option Price:[/green] ${eur_call:.4f}")
    console.print(f"[green]American Call Option Price:[/green] ${am_call:.4f}")
    console.print(f"[magenta]European Put Option Price:[/magenta] ${eur_put:.4f}")
    console.print(f"[magenta]American Put Option Price:[/magenta] ${am_put:.4f}\n")
    
    # Convergence Plot
    steps = list(range(10, 200, 10))
    prices = [binomial_tree_pricing(S, K, T, r, sigma, n, 'put', True) for n in steps]
    
    console.print("[yellow]Generating Convergence Plot...[/yellow]")
    plt.figure(figsize=(10, 6))
    plt.plot(steps, prices, marker='o', linestyle='-')
    plt.title('Convergence of American Put Option Price (Binomial Tree)')
    plt.xlabel('Number of Time Steps (N)')
    plt.ylabel('Option Price')
    plt.grid(True)
    plt.savefig('binomial_convergence.png')
    console.print("[bold green]Convergence plot saved as 'binomial_convergence.png'[/bold green]")

if __name__ == "__main__":
    main()
