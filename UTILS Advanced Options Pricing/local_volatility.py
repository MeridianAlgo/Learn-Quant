import numpy as np


def calculate_local_volatility_surface(implied_vol_matrix, strikes, maturities, spot_price=100.0, risk_free_rate=0.01):
    """
    Computes a Local Volatility Surface from an Implied Volatility Surface using the Dupire Equation.

    The discrete Dupire equation calculates local variance by applying finite differences
    to the implied volatility derivatives with respect to strike (K) and maturity (T).

    Parameters:
    implied_vol_matrix (np.ndarray): 2D array of implied volatilities (Maturities x Strikes).
    strikes (np.ndarray): 1D array of strike prices.
    maturities (np.ndarray): 1D array of time to maturities in years.
    spot_price (float): Current underlying asset price.
    risk_free_rate (float): Continuous risk free interest rate.

    Returns:
    np.ndarray: Computed local volatility matrix.
    """
    num_T, num_K = implied_vol_matrix.shape
    local_vol_matrix = np.zeros_like(implied_vol_matrix)

    for i in range(1, num_T - 1):
        for j in range(1, num_K - 1):
            T = maturities[i]
            K = strikes[j]
            iv = implied_vol_matrix[i, j]

            # Central differences for derivatives
            dT = maturities[i + 1] - maturities[i - 1]
            dK = strikes[j + 1] - strikes[j - 1]

            # First derivative with respect to Time (dT)
            dw_dT = (
                implied_vol_matrix[i + 1, j] ** 2 * maturities[i + 1]
                - implied_vol_matrix[i - 1, j] ** 2 * maturities[i - 1]
            ) / dT

            # First derivative with respect to Strike (dK)
            dw_dK = (implied_vol_matrix[i, j + 1] - implied_vol_matrix[i, j - 1]) / dK

            # Second derivative with respect to Strike (dK^2)
            dw_dK2 = (implied_vol_matrix[i, j + 1] - 2 * iv + implied_vol_matrix[i, j - 1]) / ((dK / 2) ** 2)

            # Dupire Numerator and Denominator
            numerator = dw_dT + 2 * risk_free_rate * K * dw_dK

            d1_component = (np.log(spot_price / K) + (risk_free_rate + 0.5 * iv**2) * T) / (iv * np.sqrt(T))

            denominator = K**2 * (
                dw_dK2
                - d1_component * dw_dK**2 * np.sqrt(T) / iv
                + 0.25 * dw_dK**2 * d1_component**2 * T
                + (1 / (K * iv * np.sqrt(T))) * (dw_dK * K + 1 / (iv * np.sqrt(T)))
            )

            # Prevent negative variance arising from discrete approximations mapping arbitrage violations
            variance = max(numerator / denominator, 0.0001)
            local_vol_matrix[i, j] = np.sqrt(variance)

    # Boundary conditions smoothing
    local_vol_matrix[0, :] = local_vol_matrix[1, :]
    local_vol_matrix[-1, :] = local_vol_matrix[-2, :]
    local_vol_matrix[:, 0] = local_vol_matrix[:, 1]
    local_vol_matrix[:, -1] = local_vol_matrix[:, -2]

    return local_vol_matrix


def execute_volatility_mapping():
    print("Initializing Dupire Local Volatility Engine...")

    # Mathematical grid initialization
    strikes = np.array([80, 90, 100, 110, 120])
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.25])

    # Synthetic Implied Volatility Surface displaying a recognizable "smile"
    # Options further away from the current spot of 100 have higher implied volatility
    base_iv = np.array(
        [
            [0.25, 0.22, 0.20, 0.22, 0.26],
            [0.24, 0.21, 0.19, 0.21, 0.25],
            [0.23, 0.20, 0.18, 0.20, 0.24],
            [0.22, 0.19, 0.17, 0.19, 0.23],
            [0.21, 0.18, 0.16, 0.18, 0.22],
        ]
    )

    print("\nInput Expected Implied Volatility Surface Grid (By Maturity and Strike):")
    print(base_iv)

    local_vol_surface = calculate_local_volatility_surface(
        implied_vol_matrix=base_iv, strikes=strikes, maturities=maturities, spot_price=100.0, risk_free_rate=0.02
    )

    print("\nExecuting finite difference derivations over continuous paths...")
    print("Computed Pure Local Volatility Surface Grid:")
    print(np.round(local_vol_surface, 4))
    print("\nComputation Complete. Model Ready For Exotic Product Pricing.")


if __name__ == "__main__":
    execute_volatility_mapping()
