import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "Python Basics - Comprehensions"
    / "comprehensions_tutorial.py"
)


def test_script_runs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="ignore")
    stderr = result.stderr.decode(errors="ignore")
    debug = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert result.returncode == 0, debug


def test_output_contains_comprehensions():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="ignore")
    assert "COMPREHENSION" in stdout.upper()


def test_output_contains_correlation():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="ignore")
    assert "AAPL" in stdout and "MSFT" in stdout


if __name__ == "__main__":
    test_script_runs()
    test_output_contains_comprehensions()
    test_output_contains_correlation()
    print("Comprehensions tutorial tests passed.")
