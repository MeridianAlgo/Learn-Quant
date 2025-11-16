import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'UTILS - Order Execution Simulator' / 'order_execution_simulator.py'


def test_script_runs():
    print('Testing if order_execution_simulator.py runs...')
    result = subprocess.run([sys.executable, str(SCRIPT_PATH)], input=b'6\n', capture_output=True)
    stdout = result.stdout.decode(errors='ignore') if result.stdout else ''
    stderr = result.stderr.decode(errors='ignore') if result.stderr else ''
    debug_message = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert result.returncode == 0, debug_message
    print('order_execution_simulator.py runs successfully.')

if __name__ == '__main__':
    test_script_runs()
    print('Order Execution Simulator basic test passed.')
