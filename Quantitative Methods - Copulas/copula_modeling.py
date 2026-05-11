import numpy as np
import pandas as pd
from scipy.stats import norm

class GaussianCopulaModel:
    """
    A basic implementation of a Gaussian Copula to model the dependency structure
    between two financial time series (e.g., returns of two stocks).
    """
    def __init__(self):
        self.correlation_matrix = None

    def fit(self, returns_df: pd.DataFrame):
        """
        Fit the Gaussian Copula by estimating the correlation matrix
        from the uniform marginals of the empirical data.
        """
        # Convert returns to uniform marginals using empirical CDF
        uniform_marginals = returns_df.rank(pct=True)
        
        # Transform uniform marginals to standard normal variables
        normal_marginals = uniform_marginals.apply(lambda x: norm.ppf(x.clip(lower=0.001, upper=0.999)))
        
        # Calculate Pearson correlation matrix of the transformed variables
        self.correlation_matrix = normal_marginals.corr()
        print("Fitted Gaussian Copula Correlation Matrix:")
        print(self.correlation_matrix)

    def simulate(self, num_samples: int):
        """
        Simulate new data points from the fitted copula.
        Returns simulated uniform marginals.
        """
        if self.correlation_matrix is None:
            raise ValueError("Model must be fitted before simulation.")
            
        # Simulate multivariate normal
        cov_matrix = self.correlation_matrix.values
        simulated_normals = np.random.multivariate_normal(mean=np.zeros(len(cov_matrix)), cov=cov_matrix, size=num_samples)
        
        # Transform back to uniform
        simulated_uniforms = norm.cdf(simulated_normals)
        return pd.DataFrame(simulated_uniforms, columns=self.correlation_matrix.columns)

if __name__ == "__main__":
    np.random.seed(42)
    # Generate some correlated random data
    cov = [[1, 0.8], [0.8, 1]]
    data = np.random.multivariate_normal([0, 0], cov, size=500)
    returns = pd.DataFrame(data, columns=['Asset A', 'Asset B'])
    
    copula = GaussianCopulaModel()
    copula.fit(returns)
    
    simulations = copula.simulate(10)
    print("\nSimulated Uniform Marginals:")
    print(simulations)
