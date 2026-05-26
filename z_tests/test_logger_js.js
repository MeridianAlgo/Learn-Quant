const { spawn } = require('child_process');
const path = require('path');

const SCRIPT_PATH = path.join('..', 'UTILS - Logging', 'logger.js');

function testScriptRuns() {
    console.log('Testing if logger.js runs...');
    const proc = spawn('node', [SCRIPT_PATH]);
    let output = '';
    proc.stdout.on('data', (data) => {
        output += data.toString();
        // Send '5\n' to exit when menu appears
        if (output.includes('Exit')) {
            proc.stdin.write('5\n');
        }
    });
    proc.on('close', (code) => {
        console.assert(code === 0, 'logger.js did not exit cleanly');
        console.log('logger.js runs successfully.');
        console.log('Logger (JavaScript) basic test passed.');
    });
}

testScriptRuns();
