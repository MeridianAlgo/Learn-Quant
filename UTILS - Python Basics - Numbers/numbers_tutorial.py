"""Python Numbers Tutorial for Beginners.

Run with:
    python numbers_tutorial.py

Follow along with the inline comments while this script runs. It expands on the
Level 1 material in `Documentation/Programs/level1_fundamentals.py`.
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
    print("We'll cover integers, floats, Decimal, and finance helpers.\n")


def integer_vs_float() -> None:
    """Compare integer and float operations."""
    print("=" * 60)
    print("INTEGERS VS FLOATS")
    print("=" * 60)

    trades = 7  # int
    price = 152.375  # float
    total_cost = trades * price

    print(f"Trades: {trades} (type: {type(trades).__name__})")
    print(f"Price: {price} (type: {type(price).__name__})")
    print(f"Total cost = trades * price -> {total_cost}")

    print("Floats can accumulate rounding error. Let's switch to Decimal next.")


def decimal_currency() -> None:
    """Use Decimal for precise currency math."""
    print("\n" + "=" * 60)
    print("DECIMAL FOR CURRENCY")
    print("=" * 60)

    fee = Decimal("2.49")
    balance = Decimal("1000.00")
    balance_after_fee = balance - fee

    print(f"Starting balance: {balance}")
    print(f"Fee deducted: {fee}")
    print(f"Balance after fee: {balance_after_fee}")
    print("Notice the exact cents with Decimal compared to binary floating point.")


def math_helpers() -> None:
    """Showcase built-in math helpers."""
    print("\n" + "=" * 60)
    print("MATH HELPERS")
    print("=" * 60)

    delta = -12.3456
    print(f"Absolute value of {delta}: {abs(delta)}")
    print(f"Rounded to 2 decimals: {round(delta, 2)}")
    print(f"Power function pow(2, 5): {pow(2, 5)}")


def finance_examples() -> None:
    """Demonstrate simple finance formulas."""
    print("\n" + "=" * 60)
    print("FINANCE EXAMPLES")
    print("=" * 60)

    start_price = 150
    end_price = 162
    pct_change = (end_price - start_price) / start_price
    print(f"Percent change from ${start_price} to ${end_price}: {pct_change:.2%}")

    annual_rate = 0.08
    periods = 5
    future_value = 1000 * (1 + annual_rate) ** periods
    print(f"Future value of $1000 at 8% for 5 years: ${future_value:.2f}")

    monthly_rate = (1 + annual_rate) ** (1 / 12) - 1
    print(f"Equivalent monthly rate for 8% annual: {monthly_rate:.4%}")


def main() -> None:
    intro()
    integer_vs_float()
    decimal_currency()
    math_helpers()
    finance_examples()
    print("\n🎉 Numbers tutorial complete! Continue exploring beginner utilities in the UTILS folders.")


if __name__ == "__main__":
    main()
