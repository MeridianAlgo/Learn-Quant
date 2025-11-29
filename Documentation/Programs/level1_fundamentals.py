"""Level 1: Python Fundamentals Program

This script introduces absolute beginners to Python basics. Every function below
prints *exactly* what it is doing so you can follow along in the terminal while
reading the source in `Documentation/Programs/level1_fundamentals.py`.

Concepts covered:
- Printing output
- Variables and data types
- Basic arithmetic and string operations
- Conditionals and loops
- Simple functions

Run this file directly or via the project launcher (`python main.py`).
"""

from datetime import datetime
from pathlib import Path

SOURCE_FILE = Path(__file__).resolve()


def introduction() -> None:
    """Explain how to navigate the lesson before executing examples."""
    print("\n" + "#" * 60)
    print("LEVEL 1 – PYTHON FUNDAMENTALS WALKTHROUGH")
    print("#" * 60)
    print("You're running:", SOURCE_FILE.name)
    print("Open this file in your editor to see the comments that explain each step.")
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("We'll now move through foundational Python concepts, one block at a time.")
    print("After this lesson, explore the new utilities in `UTILS - Python Basics - Strings/` and `UTILS - Python Basics - Numbers/` for extra practice.\n")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def hello_world() -> None:
    """Demonstrate printing and string concatenation."""
    print_header("Hello World")
    # The greeting is stored in a variable so learners can experiment with the value.
    name = "Python Beginner"
    print("Defining a variable called `name` with the value 'Python Beginner'.")
    print(f"Hello, {name}! Today's date is {datetime.now():%Y-%m-%d}.")


def variables_and_types() -> None:
    """Showcase common Python data types."""
    print_header("Variables and Types")
    print("We'll declare a few variables and display both their value and their type.")
    age = 21  # int
    height = 1.75  # float (meters)
    is_student = True  # bool
    favorite_language = "Python"  # str

    print(f"Age: {age} (type: {type(age).__name__})")
    print(f"Height: {height}m (type: {type(height).__name__})")
    print(f"Student status: {is_student} (type: {type(is_student).__name__})")
    print(f"Favorite language: {favorite_language} (type: {type(favorite_language).__name__})")


def arithmetic_examples() -> None:
    """Demonstrate arithmetic operations in Python."""
    print_header("Arithmetic Operators")
    print("Using integer variables `apples` and `oranges` to show math operations.")
    apples = 7
    oranges = 3
    total = apples + oranges
    difference = apples - oranges
    product = apples * oranges
    quotient = apples / oranges

    print(f"Apples + Oranges = {total}")
    print(f"Apples - Oranges = {difference}")
    print(f"Apples * Oranges = {product}")
    print(f"Apples / Oranges = {quotient:.2f}")


def control_flow() -> None:
    """Showcase if/elif/else logic and for/while loops."""
    print_header("Control Flow")
    print("Set `temperature_c` to 18 and use if/elif/else to print a message.")
    temperature_c = 18

    if temperature_c >= 25:
        print("It's warm outside.")
    elif temperature_c >= 15:
        print("It's a pleasant day.")
    else:
        print("It's a bit chilly – grab a jacket!")

    print("\nCounting with a for loop:")
    for number in range(5):
        print(f"  For loop value: {number}")

    print("\nCounting with a while loop:")
    count = 0
    while count < 3:
        print(f"  While loop value: {count}")
        count += 1


def list_operations() -> None:
    """Introduce lists and basic list methods."""
    print_header("Working with Lists")
    print("Start with a basic to-do list and practice list methods like append/remove.")
    todo_list = ["Review variables", "Practice loops", "Write a function"]
    print("Today's to-do list:")
    for item in todo_list:
        print(f"  - {item}")

    todo_list.append("Build a mini project")
    todo_list.remove("Practice loops")
    print("\nUpdated to-do list:")
    for index, item in enumerate(todo_list, start=1):
        print(f"  {index}. {item}")


def greet(name: str = "Pythonista") -> str:
    """Return a greeting message."""
    return f"Hello, {name}! You're doing great."


def functions_demo() -> None:
    """Demonstrate defining and calling functions."""
    print_header("Functions")
    print("Call the helper function `greet()` with a custom name.")
    message = greet("Learner")
    print(message)

    def square(number: int) -> int:
        """Calculate the square of a number."""
        return number * number

    for value in range(1, 4):
        print(f"Square of {value} is {square(value)}")


if __name__ == "__main__":
    introduction()
    hello_world()
    variables_and_types()
    arithmetic_examples()
    control_flow()
    list_operations()
    functions_demo()
    print("\n🎉 Great job! You've completed the Level 1 fundamentals program.")
