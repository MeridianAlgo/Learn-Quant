import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table

console = Console()

def generate_stock_returns(n_stocks=10, n_days=252):
    """
    Generate synthetic stock returns with latent factor structure.
    """
    # 2 latent factors
    market_factor = np.random.normal(0, 0.01, n_days)
    sector_factor = np.random.normal(0, 0.02, n_days)
    
    returns = []
    names = []
    
    for i in range(n_stocks):
        beta_m = np.random.uniform(0.5, 1.5)
        beta_s = np.random.uniform(-1, 1) if i < n_stocks // 2 else np.random.uniform(-0.5, 0.5)
        idiosyncratic = np.random.normal(0, 0.005, n_days)
        
        stock_ret = beta_m * market_factor + beta_s * sector_factor + idiosyncratic
        returns.append(stock_ret)
        names.append(f"Stock_{i+1}")
        
    return pd.DataFrame(np.array(returns).T, columns=names)

def main():
    console.print("[bold cyan]Principal Component Analysis for Trading[/bold cyan]")
    console.print("Demonstrating how to extract latent factors from a universe of stock returns using PCA.\n")

    console.print("[yellow]Generating synthetic stock universe returns...[/yellow]")
    df_returns = generate_stock_returns(n_stocks=20, n_days=252*2) # 2 years of data
    
    console.print("Running PCA to find orthogonal risk factors...")
    pca = PCA(n_components=5)
    pca.fit(df_returns)
    
    explained_variance = pca.explained_variance_ratio_
    
    table = Table(title="Explained Variance by Principal Components")
    table.add_column("Component", justify="center", style="cyan", no_wrap=True)
    table.add_column("Variance Explained", justify="right", style="green")
    table.add_column("Cumulative Variance", justify="right", style="magenta")
    
    cum_var = 0
    for i, var in enumerate(explained_variance):
        cum_var += var
        table.add_row(f"PC{i+1}", f"{var*100:.2f}%", f"{cum_var*100:.2f}%")
        
    console.print(table)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title('PCA Explained Variance on Stock Universe')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('pca_variance.png')
    console.print("\n[bold green]Scree plot saved as 'pca_variance.png'[/bold green]")
    
    # Analyze the weights of the first principal component (Market Portfolio)
    pc1_weights = pca.components_[0]
    
    console.print("\n[bold yellow]Component 1 Weights (often representative of the Market Factor):[/bold yellow]")
    for i, weight in enumerate(pc1_weights):
        console.print(f"Stock_{i+1}: {weight:.4f}")

if __name__ == "__main__":
    main()
