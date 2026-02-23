"""
Options Trading Strategies - Common Multi-Leg Options Strategies
"""


def bull_call_spread(stock_price, lower_strike, upper_strike, lower_premium, upper_premium):
    """
    Bull Call Spread: Buy call at lower strike, sell call at upper strike
    Limited profit, limited risk
    """
    max_profit = (upper_strike - lower_strike) - (lower_premium - upper_premium)
    max_loss = lower_premium - upper_premium
    breakeven = lower_strike + (lower_premium - upper_premium)

    # Calculate P&L at different prices
    prices = range(int(lower_strike * 0.8), int(upper_strike * 1.2), 5)
    pnl = []

    for price in prices:
        if price <= lower_strike:
            profit = -max_loss
        elif price >= upper_strike:
            profit = max_profit
        else:
            profit = (price - lower_strike) - max_loss

        pnl.append((price, profit))

    return {
        "strategy": "Bull Call Spread",
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "pnl_profile": pnl,
    }


def iron_condor(
    stock_price,
    put_lower_strike,
    put_upper_strike,
    call_lower_strike,
    call_upper_strike,
    put_lower_premium,
    put_upper_premium,
    call_lower_premium,
    call_upper_premium,
):
    """
    Iron Condor: Sell put spread + sell call spread
    Profit from low volatility
    """
    net_credit = put_upper_premium + call_lower_premium - put_lower_premium - call_upper_premium
    max_profit = net_credit
    max_loss = (put_upper_strike - put_lower_strike) - net_credit

    lower_breakeven = put_upper_strike - net_credit
    upper_breakeven = call_lower_strike + net_credit

    return {
        "strategy": "Iron Condor",
        "max_profit": max_profit,
        "max_loss": max_loss,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "net_credit": net_credit,
    }


def straddle(stock_price, strike, call_premium, put_premium):
    """
    Long Straddle: Buy call + buy put at same strike
    Profit from high volatility
    """
    total_cost = call_premium + put_premium
    upper_breakeven = strike + total_cost
    lower_breakeven = strike - total_cost

    # Calculate P&L
    prices = range(int(strike * 0.7), int(strike * 1.3), 5)
    pnl = []

    for price in prices:
        call_value = max(0, price - strike)
        put_value = max(0, strike - price)
        profit = call_value + put_value - total_cost
        pnl.append((price, profit))

    return {
        "strategy": "Long Straddle",
        "total_cost": total_cost,
        "upper_breakeven": upper_breakeven,
        "lower_breakeven": lower_breakeven,
        "pnl_profile": pnl,
    }


def butterfly_spread(
    lower_strike,
    middle_strike,
    upper_strike,
    lower_premium,
    middle_premium,
    upper_premium,
):
    """
    Butterfly Spread: Buy 1 lower, sell 2 middle, buy 1 upper
    Limited profit, limited risk
    """
    net_cost = lower_premium - 2 * middle_premium + upper_premium
    max_profit = (middle_strike - lower_strike) - net_cost
    max_loss = net_cost

    lower_breakeven = lower_strike + net_cost
    upper_breakeven = upper_strike - net_cost

    return {
        "strategy": "Butterfly Spread",
        "max_profit": max_profit,
        "max_loss": max_loss,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "net_cost": net_cost,
    }


def covered_call(stock_price, shares, strike, premium):
    """
    Covered Call: Own stock + sell call
    Generate income, limited upside
    """
    income = premium * shares
    max_profit = (strike - stock_price) * shares + income
    breakeven = stock_price - premium

    return {
        "strategy": "Covered Call",
        "income": income,
        "max_profit": max_profit,
        "breakeven": breakeven,
        "shares": shares,
    }


if __name__ == "__main__":
    print("=== Options Strategies Calculator ===\n")

    # Bull Call Spread
    result = bull_call_spread(
        stock_price=100,
        lower_strike=95,
        upper_strike=105,
        lower_premium=7,
        upper_premium=3,
    )
    print("Bull Call Spread:")
    print(f"  Max Profit: ${result['max_profit']:.2f}")
    print(f"  Max Loss: ${result['max_loss']:.2f}")
    print(f"  Breakeven: ${result['breakeven']:.2f}\n")

    # Iron Condor
    result = iron_condor(
        stock_price=100,
        put_lower_strike=90,
        put_upper_strike=95,
        call_lower_strike=105,
        call_upper_strike=110,
        put_lower_premium=1,
        put_upper_premium=3,
        call_lower_premium=3,
        call_upper_premium=1,
    )
    print("Iron Condor:")
    print(f"  Max Profit: ${result['max_profit']:.2f}")
    print(f"  Max Loss: ${result['max_loss']:.2f}")
    print(f"  Breakevens: ${result['lower_breakeven']:.2f} - ${result['upper_breakeven']:.2f}\n")

    # Straddle
    result = straddle(stock_price=100, strike=100, call_premium=5, put_premium=5)
    print("Long Straddle:")
    print(f"  Total Cost: ${result['total_cost']:.2f}")
    print(f"  Breakevens: ${result['lower_breakeven']:.2f} - ${result['upper_breakeven']:.2f}\n")

    # Covered Call
    result = covered_call(stock_price=100, shares=100, strike=105, premium=3)
    print("Covered Call:")
    print(f"  Income: ${result['income']:.2f}")
    print(f"  Max Profit: ${result['max_profit']:.2f}")
    print(f"  Breakeven: ${result['breakeven']:.2f}")
