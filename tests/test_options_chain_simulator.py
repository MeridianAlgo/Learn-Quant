import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "UTILS - Options Chain Simulator"
    / "options_chain_simulator.py"
)


def test_script_runs():
    print("Testing if options_chain_simulator.py runs...")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)], input=b"5\n", capture_output=True
    )
    stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
    stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
    debug_message = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert result.returncode == 0, debug_message
    print("options_chain_simulator.py runs successfully.")


if __name__ == "__main__":
    test_script_runs()
    print("Options Chain Simulator basic test passed.")
