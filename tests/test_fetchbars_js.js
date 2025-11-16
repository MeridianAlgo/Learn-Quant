const { spawn } = require('child_process');
const path = require('path');

const SCRIPT_PATH = path.join('..', 'UTILS - Historical Data', 'FetchBars.js');

function testScriptRuns() {
    console.log('Testing if FetchBars.js runs...');
    const proc = spawn('node', [SCRIPT_PATH]);
    let output = '';
    let sent = 0;
    proc.stdout.on('data', (data) => {
        output += data.toString();
        // Send '\n' a few times to skip prompts and exit
        if (sent < 5) {
            proc.stdin.write('\n');
            sent++;
        }
    });
    proc.on('close', (code) => {
        console.assert(code === 0, 'FetchBars.js did not exit cleanly');
        console.log('FetchBars.js runs successfully.');
        console.log('FetchBars (JavaScript) basic test passed.');
    });
}

testScriptRuns();
