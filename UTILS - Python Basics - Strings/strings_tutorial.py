"""Python Strings Tutorial for Beginners.

Run with:
    python strings_tutorial.py

Keep this file open in your editor to read the inline comments while
watching the console output. The walkthrough aligns with the Level 1
program in `Documentation/Programs/level1_fundamentals.py`.
"""

from pathlib import Path

SOURCE_FILE = Path(__file__).resolve()


def intro() -> None:
    """Explain what the learner will see."""
    print("\n" + "#" * 60)
    print("PYTHON BASICS – STRINGS WALKTHROUGH")
    print("#" * 60)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("We'll explore string creation, formatting, and common helpers.\n")


def creation_and_concatenation() -> None:
    """Demonstrate creating strings and joining pieces together."""
    print("=" * 60)
    print("CREATION & CONCATENATION")
    print("=" * 60)

    first_name = "Ada"
    last_name = "Lovelace"
    print(f"First name: {first_name}")
    print(f"Last name: {last_name}")

    full_name = first_name + " " + last_name
    print(f"Concatenated full name: {full_name}")

    repeated = "Ha" * 3
    print(f"Using * to repeat text: {repeated}")


def formatting_examples() -> None:
    """Showcase f-strings and str.format."""
    print("\n" + "=" * 60)
    print("STRING FORMATTING")
    print("=" * 60)

    ticker = "AAPL"
    price = 175.125
    change_pct = 0.0125

    print(f"f-string example: {ticker} closed at ${price:.2f} ({change_pct:.2%}).")
    print(
        "str.format example: {} closed at ${:.2f} ({:.2%}).".format(
            ticker, price, change_pct
        )
    )


def common_methods() -> None:
    """Walk through popular string helper methods."""
    print("\n" + "=" * 60)
    print("COMMON STRING METHODS")
    print("=" * 60)

    sentence = "Python makes text processing fun!"
    print(f"Original sentence: {sentence}")
    print(f"Uppercase: {sentence.upper()}")
    print(f"Lowercase: {sentence.lower()}")
    print(f"Title case: {sentence.title()}")

    words = sentence.split()
    print(f"Split into words: {words}")

    joined = "-".join(words)
    print(f"Joined with hyphen: {joined}")

    messy = "   spaced out   "
    print(f"Before strip(): '{messy}' -> After strip(): '{messy.strip()}'")


def validation_examples() -> None:
    """Demonstrate basic checks used in data cleaning."""
    print("\n" + "=" * 60)
    print("VALIDATION & CLEANING")
    print("=" * 60)

    sample_email = "user@example.com"
    print(f"Does '{sample_email}' contain '@'?", "@" in sample_email)
    print("Does it end with '.com'?", sample_email.endswith(".com"))

    filename = "report.csv"
    if not filename.endswith(".csv"):
        print("Only CSV files accepted.")
    else:
        print(f"'{filename}' is a valid CSV file name.")


def mini_project() -> None:
    """Convert a comma-separated string into cleaned tickers."""
    print("\n" + "=" * 60)
    print("MINI PROJECT – CLEAN TICKER INPUT")
    print("=" * 60)

    raw_input = " AAPL, msft , GOOGL, amzn "
    print(f"Raw input: '{raw_input}'")

    cleaned = [
        ticker.strip().upper() for ticker in raw_input.split(",") if ticker.strip()
    ]
    print(f"Cleaned tickers: {cleaned}")

    if cleaned:
        print("Great! You can now feed this list into other utilities.")


def main() -> None:
    intro()
    creation_and_concatenation()
    formatting_examples()
    common_methods()
    validation_examples()
    mini_project()
    print(
        "\n🎉 Strings tutorial complete! Continue exploring Python Basics in the UTILS folders."
    )


if __name__ == "__main__":
    main()
