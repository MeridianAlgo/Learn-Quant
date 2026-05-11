"""Command-line interface for the interactive Python learning platform."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:  # Ensure local imports work via runpy
    sys.path.insert(0, str(PACKAGE_ROOT))

# Module-level imports must be at top, but we need to modify sys.path first
# This is a valid exception to E402 for dynamic path modification
from content import LESSONS, Lesson, LessonSection, QuizQuestion, get_lesson_by_slug, list_lessons  # noqa: E402

BORDER = "=" * 80
SOURCE_FILE = Path(__file__).resolve()


def print_banner() -> None:
    print(BORDER)
    print("PYTHON LEARNING PLATFORM – INTERACTIVE CLI")
    print(BORDER)
    print(f"Source: {SOURCE_FILE.relative_to(PACKAGE_ROOT.parent)}")
    print("Browse lessons, read guided sections, and take short quizzes.\n")


def main_menu() -> None:
    print("Available lessons:\n")
    for index, lesson in enumerate(list_lessons(), start=1):
        print(f"  {index}. {lesson['title']} ({lesson['difficulty']}, {lesson['estimated_minutes']} min)")
        print(f"     ↳ {lesson['summary']}")
    print("\nCommands:")
    print("  [number] → open lesson")
    print("  L        → list lessons again")
    print("  Q        → quit")


def prompt(message: str) -> str:
    try:
        return input(message)
    except EOFError:  # pragma: no cover - interactive fallback
        print()
        return ""


def choose_lesson() -> Lesson | None:
    while True:
        choice = prompt("Select a lesson (number or slug, Q to quit): ").strip()
        if not choice:
            continue
        if choice.lower() == "q":
            return None
        if choice.lower() == "l":
            main_menu()
            continue
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(LESSONS):
                return LESSONS[index]
            print("Invalid lesson number. Try again.")
            continue
        lesson = get_lesson_by_slug(choice)
        if lesson:
            return lesson
        print("No lesson found with that identifier. Try again.")


def render_section(section: LessonSection) -> None:
    print("\n" + BORDER)
    print(section.title.upper())
    print(BORDER)
    print(section.body)
    if section.code_example:
        print("\nExample code:")
        print("-" * 40)
        print(section.code_example)
        print("-" * 40)
    if section.quiz:
        run_quiz(section.quiz)


def run_quiz(questions: Iterable[QuizQuestion]) -> None:
    for number, question in enumerate(questions, start=1):
        print(f"\nQuiz {number}: {question.prompt}")
        for idx, choice in enumerate(question.choices, start=1):
            print(f"  {idx}. {choice}")
        while True:
            answer = prompt("Your answer (number): ").strip()
            if not answer:
                continue
            if not answer.isdigit():
                print("Please enter a choice number.")
                continue
            response_idx = int(answer) - 1
            if response_idx == question.answer_index:
                print("✅ Correct! " + question.explanation)
                break
            print("❌ Not quite. " + question.explanation)
            try_again = prompt("Try again? (y/n): ").strip().lower()
            if try_again != "y":
                break


def print_objectives(lesson: Lesson) -> None:
    print("\nLearning objectives:")
    for bullet in lesson.objectives:
        print(f"  • {bullet}")


def print_practice(lesson: Lesson) -> None:
    print("\nPractice prompts:")
    for idx, item in enumerate(lesson.practice, start=1):
        print(f"  {idx}. {item}")


def print_follow_up(lesson: Lesson) -> None:
    print("\nSuggested follow-up resources:")
    for resource in lesson.follow_up:
        print(f"  → {resource}")


def lesson_loop(lesson: Lesson) -> None:
    print("\n" + BORDER)
    print(f"LESSON: {lesson.title} ({lesson.difficulty})")
    print(BORDER)
    print(f"Estimated time: {lesson.estimated_minutes} minutes")
    print(f"Summary: {lesson.summary}")
    print_objectives(lesson)

    for section in lesson.sections:
        prompt("\nPress Enter to open the next section...")
        render_section(section)

    print_practice(lesson)
    print_follow_up(lesson)
    prompt("\nPress Enter to return to the main menu...")


def guided_walkthrough() -> None:
    print_banner()
    main_menu()
    while True:
        lesson = choose_lesson()
        if lesson is None:
            print("\nHappy learning! ✨")
            return
        lesson_loop(lesson)
        main_menu()


if __name__ == "__main__":
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    guided_walkthrough()
