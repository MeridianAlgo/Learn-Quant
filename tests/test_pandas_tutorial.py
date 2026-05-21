import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "Python Basics - Pandas" / "pandas_tutorial.py"


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


def test_output_contains_resampling():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="ignore")
    assert "RESAMPLING" in stdout or "weekly" in stdout.lower()


def test_output_contains_signal():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="ignore")
    assert "signal" in stdout.lower()


if __name__ == "__main__":
    test_script_runs()
    test_output_contains_resampling()
    test_output_contains_signal()
    print("Pandas tutorial tests passed.")
