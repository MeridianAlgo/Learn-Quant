import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "Python Basics - NumPy" / "numpy_tutorial.py"


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
    assert "NumPy" in stdout or "numpy" in stdout.lower(), debug


def test_output_contains_sharpe():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="ignore")
    assert "Sharpe" in stdout


def test_output_contains_portfolio_vol():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="ignore")
    assert "portfolio" in stdout.lower()


if __name__ == "__main__":
    test_script_runs()
    test_output_contains_sharpe()
    test_output_contains_portfolio_vol()
    print("NumPy tutorial tests passed.")
