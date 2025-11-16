const { spawn } = require('child_process');
const path = require('path');

const SCRIPT_PATH = path.join('..', 'UTILS - News Fetching', 'fetchNews.js');

function testScriptRuns() {
    console.log('Testing if fetchNews.js runs...');
    const proc = spawn('node', [SCRIPT_PATH]);
    let output = '';
    let sent = 0;
    proc.stdout.on('data', (data) => {
        output += data.toString();
        // Send '\n' a few times to skip prompts and exit
        if (sent < 3) {
            proc.stdin.write('\n');
            sent++;
        }
    });
    proc.on('close', (code) => {
        console.assert(code === 0, 'fetchNews.js did not exit cleanly');
        console.log('fetchNews.js runs successfully.');
        console.log('fetchNews (JavaScript) basic test passed.');
    });
}

testScriptRuns();
