"""Lesson content and helpers for the interactive learning platform."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class QuizQuestion:
    prompt: str
    choices: List[str]
    answer_index: int
    explanation: str


@dataclass(frozen=True)
class LessonSection:
    title: str
    body: str
    code_example: Optional[str] = None
    quiz: Optional[List[QuizQuestion]] = None


@dataclass(frozen=True)
class Lesson:
    slug: str
    title: str
    difficulty: str
    estimated_minutes: int
    summary: str
    objectives: List[str]
    sections: List[LessonSection]
    practice: List[str]
    follow_up: List[str]


def _body(text: str) -> str:
    """Utility to de-indent multi-line prose."""
    return dedent(text).strip()


LESSONS: List[Lesson] = [
    Lesson(
        slug="python-foundations",
        title="Python Foundations",
        difficulty="Beginner",
        estimated_minutes=25,
        summary="Learn how Python executes code, manages variables, and prints descriptive output.",
        objectives=[
            "Understand how to run scripts from the CLI or launcher",
            "Differentiate between numbers, strings, and booleans",
            "Practice using f-strings to produce readable guidance",
        ],
        sections=[
            LessonSection(
                title="Running Your First Script",
                body=_body(
                    """
                    Python files execute top-to-bottom. Each of the utilities in this repository prints
                    context explaining *what* it is doing and *why*. When you run a program, narrate its
                    behavior to the learner. Try printing an introduction that names the file, the folder,
                    and the key idea demonstrated in the next block.
                    """
                ),
                code_example=dedent(
                    """
                    from pathlib import Path

                    SOURCE = Path(__file__).name
                    print(f"👋 Welcome! You're running {SOURCE}.")
                    print("We'll store a learner name and greet them using formatted strings.")
                    """
                ).strip(),
            ),
            LessonSection(
                title="Working with Variables",
                body=_body(
                    """
                    Variables are *labels* pointing at values in memory. Use descriptive names that tell a story.
                    Show the learner both the value and the type to reinforce understanding. Encourage them to
                    modify values and re-run to see updated output.
                    """
                ),
                code_example=dedent(
                    """
                    learner_name = "Avery"
                    study_minutes = 40
                    is_motivated = True

                    print(f"Learner: {learner_name} (type: {type(learner_name).__name__})")
                    print(f"Study minutes today: {study_minutes}")
                    print(f"Motivated? {is_motivated}")
                    """
                ).strip(),
                quiz=[
                    QuizQuestion(
                        prompt="Which statement accurately describes a Python variable?",
                        choices=[
                            "A reserved keyword that cannot change",
                            "A label that refers to a value stored in memory",
                            "A data type limited to numbers",
                            "A function that prints to the console",
                        ],
                        answer_index=1,
                        explanation="Variables are names that reference values. The same name can point to different values over time.",
                    )
                ],
            ),
            LessonSection(
                title="Formatting Output",
                body=_body(
                    """
                    F-strings are the go-to tool for writing clean, instructional output. Surround expressions
                    with braces `{}` inside an f-string to interpolate values. You can also format numbers with
                    precision specifiers, like `:.2f` for two decimal places.
                    """
                ),
                code_example=dedent(
                    """
                    portfolio_value = 10543.876
                    daily_change = 0.0325

                    print(f"Your portfolio is worth ${portfolio_value:,.2f}.")
                    print(f"That is a {daily_change:.2%} change compared to yesterday.")
                    """
                ).strip(),
            ),
        ],
        practice=[
            "Ask the learner to input their name with `input()` and print a personalized greeting.",
            "Track study goals using two variables: `target_minutes` and `minutes_completed`.",
            "Format a summary line that combines strings, integers, and booleans.",
        ],
        follow_up=[
            "Run `UTILS - Python Basics - Strings/strings_tutorial.py` for deeper string practice.",
            "Explore `Documentation/Programs/level1_fundamentals.py` to see these concepts in action.",
        ],
    ),
    Lesson(
        slug="control-flow",
        title="Control Flow & Decisions",
        difficulty="Beginner",
        estimated_minutes=30,
        summary="Guide learners through conditionals and loops with real-world, finance-inspired examples.",
        objectives=[
            "Decide when to use `if/elif/else` versus `match` statements",
            "Trace loop execution with printed narration",
            "Build small decision helpers, like trade eligibility checks",
        ],
        sections=[
            LessonSection(
                title="Conditionals Tell Stories",
                body=_body(
                    """
                    Decision blocks should read like short stories. State the *condition*, the *action*, and
                    the *reasoning*. Encourage learners to write branch outputs that narrate the outcome.
                    """
                ),
                code_example=dedent(
                    """
                    price = 132.5
                    threshold = 130
                    if price >= threshold:
                        print("✅ Price met our entry condition. Submit the simulated order.")
                    else:
                        print("⌛ Still waiting. Explain what to monitor next.")
                    """
                ).strip(),
            ),
            LessonSection(
                title="Looping with Intention",
                body=_body(
                    """
                    Use loops to *process* collections or to *repeat* actions under specific conditions.
                    Narrate the iteration to help learners follow the flow. Show both `for` and `while`
                    loops and explain when each is appropriate.
                    """
                ),
                code_example=dedent(
                    """
                    tasks = ["Collect data", "Compute indicators", "Generate trade idea"]
                    for step_number, task in enumerate(tasks, start=1):
                        print(f"Step {step_number}: {task}")

                    countdown = 3
                    while countdown > 0:
                        print(f"Launching backtest in {countdown}...")
                        countdown -= 1
                    """
                ).strip(),
                quiz=[
                    QuizQuestion(
                        prompt="When should you prefer a `while` loop over a `for` loop?",
                        choices=[
                            "When you know the number of iterations in advance",
                            "When iterating over each item in a list",
                            "When the loop should continue until a condition changes",
                            "Never; `for` loops replace `while` loops",
                        ],
                        answer_index=2,
                        explanation="`while` loops are ideal when the number of iterations depends on a condition evaluated during execution.",
                    )
                ],
            ),
            LessonSection(
                title="Mini Project: Risk Check",
                body=_body(
                    """
                    Combine conditionals and loops to build a reusable helper. For example, verify whether a
                    trade fits within a risk budget. Provide plenty of commentary so learners can adapt the
                    template to their own strategies.
                    """
                ),
                code_example=dedent(
                    """
                    def within_risk_budget(position_size: int, max_risk: int) -> bool:
                        print(f"Checking if {position_size} shares exceeds the max risk of {max_risk} shares...")
                        if position_size <= max_risk:
                            print("Position approved ✅")
                            return True
                        print("Position rejected ❌")
                        return False

                    for size in [50, 120, 80]:
                        within_risk_budget(size, max_risk=100)
                    """
                ).strip(),
            ),
        ],
        practice=[
            "Ask the learner to categorize portfolio volatility (`low`, `medium`, `high`) using thresholds.",
            "Create a quiz that loops until the learner answers correctly, explaining the reasoning each time.",
        ],
        follow_up=[
            "Review loops in `Documentation/Programs/level1_fundamentals.py`.",
            "Experiment with `UTILS - Data Structures - Lists/lists.py` to iterate over more complex datasets.",
        ],
    ),
    Lesson(
        slug="data-structures",
        title="Core Data Structures",
        difficulty="Intermediate",
        estimated_minutes=35,
        summary="Dive deeper into lists, dictionaries, and simple NumPy arrays focused on financial analysis.",
        objectives=[
            "Differentiate when to use lists versus dictionaries",
            "Interpret tabular data using simple pandas DataFrames",
            "Introduce learners to NumPy arrays for vectorized operations",
        ],
        sections=[
            LessonSection(
                title="Lists vs. Dictionaries",
                body=_body(
                    """
                    Introduce real datasets: watchlists, order books, or cash balances. Lists maintain order and
                    allow duplicates, while dictionaries offer quick lookups. Provide examples learners can edit.
                    """
                ),
                code_example=dedent(
                    """
                    watchlist = ["AAPL", "MSFT", "NVDA"]
                    balances = {"USD": 5400.25, "EUR": 2100.10}

                    print("Watchlist symbols:")
                    for symbol in watchlist:
                        print(f"  - {symbol}")

                    print("\nCurrency balances:")
                    for currency, amount in balances.items():
                        print(f"  {currency}: {amount:,.2f}")
                    """
                ).strip(),
            ),
            LessonSection(
                title="Pandas Snapshots",
                body=_body(
                    """
                    Showcase a small DataFrame summarizing prices and volatility. Emphasize descriptive statistics
                    and sorting logic. Encourage learners to tweak column names or add indicators.
                    """
                ),
                code_example=dedent(
                    """
                    import pandas as pd

                    data = {
                        "symbol": ["AAPL", "MSFT", "GOOGL"],
                        "price": [175.12, 402.15, 125.39],
                        "volatility": [0.022, 0.018, 0.025],
                    }
                    df = pd.DataFrame(data)
                    print(df)
                    print("\nSorted by volatility:")
                    print(df.sort_values("volatility"))
                    """
                ).strip(),
            ),
            LessonSection(
                title="Intro to NumPy Arrays",
                body=_body(
                    """
                    NumPy arrays enable fast math for portfolios. Demonstrate element-wise arithmetic and simple
                    aggregations. Connect results to financial interpretations, such as cumulative returns.
                    """
                ),
                code_example=dedent(
                    """
                    import numpy as np

                    returns = np.array([0.01, -0.004, 0.012])
                    print("Mean return:", returns.mean())
                    print("Cumulative return:", (1 + returns).prod() - 1)
                    """
                ).strip(),
            ),
        ],
        practice=[
            "Convert the `watchlist` into a list of dictionaries that stores latest price and sector.",
            "Add a column to the DataFrame for percentage weight and normalize it to 100%.",
            "Use NumPy to annualize the provided daily returns (multiply mean by 252).",
        ],
        follow_up=[
            "Jump to `Documentation/Programs/level2_intermediate.py` to see these structures in project form.",
            "Study `UTILS - Quantitative Methods - Time Series/time_series_tools.py` for applied pandas workflows.",
        ],
    ),
    Lesson(
        slug="financial-storytelling",
        title="Financial Storytelling",
        difficulty="Intermediate",
        estimated_minutes=30,
        summary="Teach learners how to narrate insights from calculations and data structures they just built.",
        objectives=[
            "Translate numbers into plain-language narratives",
            "Guide learners through journaling their coding sessions",
            "Showcase how to connect beginner utilities to advanced finance modules",
        ],
        sections=[
            LessonSection(
                title="From Numbers to Narratives",
                body=_body(
                    """
                    After computing metrics, summarize *what they mean*. Encourage learners to write journal entries
                    describing the insight gained from each calculation. Provide a template they can reuse.
                    """
                ),
                code_example=dedent(
                    """
                    sharpe_ratio = 0.72
                    explanation = (
                        "Sharpe above 0.5 suggests our strategy delivered excess return relative to risk. "
                        "Next, compare it with the benchmark to decide if rebalancing is necessary."
                    )
                    print(f"Sharpe Ratio: {sharpe_ratio:.2f}\n{explanation}")
                    """
                ).strip(),
            ),
            LessonSection(
                title="Journaling Template",
                body=_body(
                    """
                    Encourage reflective practice. Provide a simple note-taking format where learners log date,
                    objective, what they built, and what they would try next time. Storing this in a text file makes
                    it easy to review progress.
                    """
                ),
                code_example=dedent(
                    """
                    from datetime import date

                    def journal_entry(task: str, takeaway: str, next_step: str) -> str:
                        return (
                            f"Date: {date.today()}\n"
                            f"Task: {task}\n"
                            f"Key takeaway: {takeaway}\n"
                            f"Next step: {next_step}\n"
                        )

                    note = journal_entry(
                        task="Computed SMA crossover",
                        takeaway="Understood how window sizes affect signal lag.",
                        next_step="Explore RSI to complement the trend signal.",
                    )
                    print(note)
                    """
                ).strip(),
            ),
        ],
        practice=[
            "Summarize the story behind a volatility spike in plain English.",
            "Write a journal entry after completing another lesson in this platform.",
        ],
        follow_up=[
            "Connect to `Documentation/Programs/level3_financial.py` to narrate portfolio analytics.",
            "Practice sentiment storytelling with `UTILS - News Fetching/fetchNews.js` output.",
        ],
    ),
]


def list_lessons() -> Iterable[dict]:
    """Return metadata about the available lessons."""
    for lesson in LESSONS:
        yield {
            "slug": lesson.slug,
            "title": lesson.title,
            "difficulty": lesson.difficulty,
            "estimated_minutes": lesson.estimated_minutes,
            "summary": lesson.summary,
        }


def get_lesson_by_slug(slug: str) -> Optional[Lesson]:
    """Return a lesson for the provided slug (case-insensitive)."""
    slug_lower = slug.lower()
    for lesson in LESSONS:
        if lesson.slug.lower() == slug_lower:
            return lesson
    return None
