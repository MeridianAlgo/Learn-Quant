"""
Discounted Cash Flow (DCF) Calculator
---------------------------------------
DCF is one of the key methods for valuing stocks, businesses, projects, and investments!
"""


def discounted_cash_flow(future_cash_flows, discount_rate):
    """
    Calculate the present value of future cash flows using DCF.
    Args:
        future_cash_flows (list of float): Cash flows for each period (e.g., years)
        discount_rate (float): Discount rate per period (as decimal, e.g., 0.1 for 10%)
    Returns:
        float: Net present value (NPV) of the future cash flows
    """
    npv = 0
    for t, cash in enumerate(future_cash_flows, 1):
        npv += cash / (1 + discount_rate) ** t
    return npv


if __name__ == "__main__":
    # Example: Value a project with cash flows over 4 years
    flows = [1000, 1200, 1500, 2000]
    discount = 0.08  # 8% per year
    present_value = discounted_cash_flow(flows, discount)
    print(f"Present Value of Project: ${present_value:,.2f}")
