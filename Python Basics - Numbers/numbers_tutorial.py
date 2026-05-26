"""Python Numbers Tutorial for Beginners.

LEARNING PATH: This file teaches Python number fundamentals used in quantitative finance.
Work through this from top to bottom while keeping the file open in your editor.

Run with:
    python numbers_tutorial.py

Keep this file open as you run it—watch the console output while reading the comments
in the source code. It expands on Level 1 material in
`Documentation/Programs/level1_fundamentals.py`.

WHY THIS MATTERS:
- Integers and floats are the foundation of all numerical computing
- In finance, precision matters: using the wrong number type costs real money
- This tutorial shows you WHEN to use int, float, or Decimal based on real scenarios
"""

from decimal import Decimal, getcontext
from pathlib import Path

SOURCE_FILE = Path(__file__).resolve()
getcontext().prec = 6


def intro() -> None:
    """Print orientation details for the learner."""
    print("\n" + "#" * 60)
    print("PYTHON BASICS – NUMBERS WALKTHROUGH")
    print("#" * 60)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("We'll cover integers, floats, Decimal, and real finance helpers.\n")


def integer_vs_float() -> None:
    """Compare integer and float operations.

    GOAL: Understand when Python stores numbers as integers (whole numbers) vs
    floats (decimals), and why this matters.

    KEY INSIGHT:
    - Use INT when: counting shares, number of trades, or anything that can't be fractional
    - Use FLOAT when: prices, returns, percentages (but watch out for rounding errors!)
    """
    print("=" * 60)
    print("INTEGERS VS FLOATS")
    print("=" * 60)

    # Integers: whole numbers, no decimal point
    trades = 7  # We execute exactly 7 trades
    price = 152.375  # Price in dollars
    total_cost = trades * price

    print(f"Number of trades: {trades} (type: {type(trades).__name__})")
    print(f"  -> An integer (can't do 7.5 trades)")

    print(f"\nPrice per share: ${price} (type: {type(price).__name__})")
    print(f"  -> A float (prices have decimals)")

    print(f"\nTotal cost = {trades} × ${price}")
    print(f"  = {total_cost} (type: {type(total_cost).__name__})")

    # WARNING: Floats can accumulate rounding errors
    print("\nWARNING - FLOAT ROUNDING ERROR:")
    result = 0.1 + 0.2
    print(f"Try this: 0.1 + 0.2 = {result}")
    print(f"Is it exactly 0.3? {result == 0.3}")
    print("Why? Floats use binary representation—0.1 can't be represented exactly")
    print("This is why we use Decimal for money!")


def decimal_currency() -> None:
    """Use Decimal for precise currency math.

    GOAL: Learn why Decimal is required for money, not optional.

    KEY INSIGHT:
    When dealing with money, precision matters. Using float means you might calculate
    $100.00 but get 100.00000000001 internally. In large trading systems, this
    compounds into thousands of dollars in errors. Use Decimal from the start!
    """
    print("\n" + "=" * 60)
    print("DECIMAL FOR CURRENCY")
    print("=" * 60)

    # ALWAYS create Decimal from a STRING, not a float!
    fee = Decimal("2.49")
    balance = Decimal("1000.00")
    balance_after_fee = balance - fee

    print(f"Starting balance: ${balance}")
    print(f"Trading fee deducted: ${fee}")
    print(f"Balance after fee: ${balance_after_fee}")

    print("\nFLOAT VS DECIMAL COMPARISON:")
    balance_float = 1000.00
    fee_float = 2.49
    balance_after_fee_float = balance_float - fee_float

    print(f"Using float:   ${balance_after_fee_float:.20f}")
    print(f"Using Decimal: ${balance_after_fee}")

    # Demonstrate compound error with multiple transactions
    print("\nCOMPOUND ERROR WITH 100 SMALL TRANSACTIONS:")

    balance_f = 1000.0
    for _ in range(100):
        balance_f = balance_f * 0.99  # Each transaction is 1% fee

    balance_d = Decimal("1000.00")
    rate_d = Decimal("0.99")
    for _ in range(100):
        balance_d = balance_d * rate_d

    print(f"Float result:   ${balance_f:.10f}")
    print(f"Decimal result: ${balance_d}")
    print(f"Difference: ${float(balance_d) - balance_f:.10f}")
    print("See how Decimal keeps precision while float drifts!")


def math_helpers() -> None:
    """Showcase built-in math helpers and when to use them.

    GOAL: Learn standard functions Python provides for number manipulation,
    and when they're useful in finance.

    FUNCTIONS COVERED:
    - abs(): absolute value
    - round(): round to N decimal places
    - pow(): raise to a power (used in compound calculations)
    """
    print("\n" + "=" * 60)
    print("MATH HELPERS")
    print("=" * 60)

    # abs() returns absolute value (removes negative sign)
    price_change = -12.3456
    magnitude = abs(price_change)
    print(f"Price change: ${price_change} (lost money)")
    print(f"Magnitude of loss: ${magnitude} (absolute value)")

    # round() rounds to N decimal places
    print(f"\nRounded to 2 decimals: {round(price_change, 2)}")
    print(f"(This is what you'd show a customer)")

    # pow(base, exponent) calculates base^exponent
    print(f"\nPower function pow(2, 5) = 2^5 = {pow(2, 5)}")
    print(f"(Useful for compound interest, exponential growth)")


def finance_examples() -> None:
    """Demonstrate simple finance formulas using Python math.

    GOAL: Show how the number concepts above combine into real finance calculations.

    FORMULAS USED:
    1. Percentage change = (end - start) / start
    2. Future value = present_value * (1 + rate)^periods
    3. Converting annual rate to monthly = (1 + annual)^(1/12) - 1
    """
    print("\n" + "=" * 60)
    print("FINANCE EXAMPLES")
    print("=" * 60)

    # EXAMPLE 1: Percent Change
    print("\n[1] PERCENT CHANGE (Return Calculation)")
    print("-" * 40)

    start_price = 150
    end_price = 162
    pct_change = (end_price - start_price) / start_price

    print(f"Stock bought at: ${start_price}")
    print(f"Stock sold at:   ${end_price}")
    print(f"Return = ({end_price} - {start_price}) / {start_price}")
    print(f"       = {pct_change:.4f} (as decimal)")
    print(f"       = {pct_change:.2%} (as percentage)")

    # EXAMPLE 2: Compound Interest
    print("\n[2] COMPOUND INTEREST (Future Value)")
    print("-" * 40)

    principal = 1000.0
    annual_rate = 0.08
    years = 5

    future_value = principal * (1 + annual_rate) ** years

    print(f"Initial investment: ${principal:.2f}")
    print(f"Annual interest rate: {annual_rate:.1%}")
    print(f"Time period: {years} years")
    print(f"\nFuture Value = ${principal} × (1.08)^{years}")
    print(f"            = ${future_value:.2f}")
    print(f"Gain: ${future_value - principal:.2f}")

    # EXAMPLE 3: Converting Annual Rate to Monthly
    print("\n[3] TIME CONVERSION (Annual to Monthly Rate)")
    print("-" * 40)

    monthly_rate = (1 + annual_rate) ** (1 / 12) - 1

    print(f"Annual rate: {annual_rate:.2%}")
    print(f"Naive calculation: {annual_rate / 12:.4%} (WRONG!)")
    print(f"Correct calculation: (1.08)^(1/12) - 1")
    print(f"                   = {monthly_rate:.4%}")
    print(f"\nWhy the difference? Because interest compounds!")
    print(f"If you earned {monthly_rate:.4%} monthly for 12 months:")
    actual_annual = (1 + monthly_rate) ** 12 - 1
    print(f"  You'd earn {actual_annual:.2%} per year (exactly 8%)")
    print(f"But if you used {annual_rate / 12:.4%} monthly:")
    wrong_annual = (1 + annual_rate / 12) ** 12 - 1
    print(f"  You'd earn {wrong_annual:.2%} per year (only ~7.9%)")


def main() -> None:
    """Execute the full walkthrough in order."""
    intro()
    integer_vs_float()
    decimal_currency()
    math_helpers()
    finance_examples()

    print("\n" + "=" * 60)
    print("NUMBERS TUTORIAL COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Use INT for counts, FLOAT for measurements")
    print("2. Use DECIMAL for money (always!)")
    print("3. Use pow() for compound calculations")
    print("4. Remember: (1 + rate)^(1/12) not rate/12")
    print("\nYou're ready for:")
    print("  -> Python Basics - Strings")
    print("  -> Data Structures - Arrays")
    print("  -> Quantitative Methods - Statistics")
    print("\nHappy learning!")


if __name__ == "__main__":
    main()
