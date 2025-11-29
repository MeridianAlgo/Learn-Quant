"""
Bond Price and Yield Calculator
-------------------------------
Simple tools for finding the present value (price) of a plain-vanilla bond, and estimating yield to maturity (YTM).
"""
def bond_price(face_value, coupon_rate, periods, ytm):
    """
    Calculate the price of a fixed-rate bond.
    Args:
        face_value (float): How much the bond will pay at maturity (e.g., $1000)
        coupon_rate (float): Annual coupon rate (as decimal, e.g. 0.05)
        periods (int): Number of periods until maturity (e.g., years)
        ytm (float): Yield to maturity (as decimal)
    Returns:
        float: Bond price
    """
    price = 0
    for t in range(1, periods + 1):
        price += (face_value * coupon_rate) / (1 + ytm) ** t
    price += face_value / (1 + ytm) ** periods
    return price

def estimate_ytm(face_value, coupon_rate, periods, price, tol=1e-5):
    """
    Estimate yield to maturity (YTM) using binary search (for beginners).
    Args:
        face_value (float): Face value
        coupon_rate (float): Coupon rate
        periods (int): Number of periods
        price (float): Observed bond price
        tol (float): Solution tolerance
    Returns:
        float: Estimated YTM
    """
    low, high = 0, 1
    while high - low > tol:
        mid = (low + high) / 2
        guess_price = bond_price(face_value, coupon_rate, periods, mid)
        if guess_price > price:
            low = mid
        else:
            high = mid
    return (low + high) / 2

if __name__ == "__main__":
    # Example: Price a 3-year, 5% coupon bond, $1000 face, YTM 6%
    p = bond_price(1000, 0.05, 3, 0.06)
    ytm = estimate_ytm(1000, 0.05, 3, p)
    print(f"Bond Price: {p:.2f}, Estimated YTM: {ytm:.4f}")
