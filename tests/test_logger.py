import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "UTILS - Logging" / "logger.py"


def test_script_runs():
    print("Testing if logger.py runs...")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)], input=b"5\n", capture_output=True
    )
    stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
    stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
    debug_message = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert result.returncode == 0, debug_message
    print("logger.py runs successfully.")


if __name__ == "__main__":
    test_script_runs()
    print("Logger (Python) basic test passed.")
