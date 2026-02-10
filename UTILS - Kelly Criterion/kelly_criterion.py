"""
Kelly Criterion Implementation for optimal position sizing in trading.
"""


def calculate_kelly_fraction(win_probability, win_loss_ratio):
    """
    Calculates the Kelly fraction (f*) for a single bet/trade.

    Args:
        win_probability (float): Probability of a winning trade (p).
        win_loss_ratio (float): Ratio of average win to average loss (b).

    Returns:
        float: The optimal fraction of the bankroll to risk.
    """
    p = win_probability
    q = 1 - p
    b = win_loss_ratio

    if b <= 0:
        return 0.0

    kelly_f = (b * p - q) / b
    return max(0.0, kelly_f)


def calculate_half_kelly(kelly_f):
    """
    Calculates 'Half Kelly' which is often used to reduce volatility.
    """
    return kelly_f / 2.0


if __name__ == "__main__":
    # Example 1: High win rate, lower win/loss ratio
    p1 = 0.60  # 60% win rate
    b1 = 1.0  # 1:1 reward to risk
    f1 = calculate_kelly_fraction(p1, b1)

    # Example 2: Lower win rate, higher win/loss ratio
    p2 = 0.35  # 35% win rate
    b2 = 3.0  # 3:1 reward to risk
    f2 = calculate_kelly_fraction(p2, b2)

    print("Kelly Criterion Examples")
    print("-" * 30)
    print(f"Strategy 1 (p={p1}, b={b1}): Optimal f* = {f1:.2%}")
    print(f"Strategy 1 Half Kelly: {calculate_half_kelly(f1):.2%}")
    print(f"Strategy 2 (p={p2}, b={b2}): Optimal f* = {f2:.2%}")
    print(f"Strategy 2 Half Kelly: {calculate_half_kelly(f2):.2%}")
