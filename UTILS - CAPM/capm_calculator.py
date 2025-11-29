"""
CAPM (Capital Asset Pricing Model) Utility
------------------------------------------
This script calculates the expected return of an investment using CAPM, one of the most iconic models in finance.
"""
def capm_expected_return(risk_free_rate, beta, market_return):
    """
    Calculate the expected return using CAPM.
    Args:
        risk_free_rate (float): Risk-free rate as a decimal (e.g. 0.03 for 3%)
        beta (float): Beta of the asset (how much it moves relative to market)
        market_return (float): Expected market return as decimal
    Returns:
        float: Expected return using CAPM
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)

if __name__ == "__main__":
    # Example usage
    rf = 0.03  # 3% risk-free rate
    beta = 1.2  # Asset is 20% more volatile than market
    rm = 0.09  # Expected market return 9%
    exp_ret = capm_expected_return(rf, beta, rm)
    print(f"Expected CAPM Return: {exp_ret:.2%}")
