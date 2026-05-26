import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "UTILS - AI Development" / "chatbot.py"


def test_script_runs():
    print("Testing if chatbot.py runs...")
    result = subprocess.run([sys.executable, str(SCRIPT_PATH)], input=b"exit\n", capture_output=True)
    assert result.returncode == 0
    print("chatbot.py runs successfully.")


if __name__ == "__main__":
    test_script_runs()
    print("Chatbot (Python) basic test passed.")
