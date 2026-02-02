import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "UTILS - Websocket Connection" / "finnhub.py"
)


@pytest.mark.skipif(
    not os.getenv("FINNHUB_API_KEY"),
    reason="FINNHUB_API_KEY environment variable not set; skipping live websocket test",
)
def test_script_runs():
    assert SCRIPT_PATH.exists(), f"Missing script at {SCRIPT_PATH}"
    print("Testing if finnhub.py runs...")
    env = os.environ.copy()
    env.setdefault("FINNHUB_API_KEY", "FINNHUB_API_KEY")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)], capture_output=True, env=env
    )
    stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
    stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
    debug_message = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert result.returncode == 0, debug_message
    print("finnhub.py runs successfully.")


if __name__ == "__main__":
    test_script_runs()
    print("Finnhub Websocket basic test passed.")
