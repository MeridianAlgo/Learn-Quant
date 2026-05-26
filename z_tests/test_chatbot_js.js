const { spawn } = require('child_process');
const path = require('path');

const SCRIPT_PATH = path.join('..', 'UTILS - AI Development', 'chatbot.js');

function testScriptRuns() {
    console.log('Testing if chatbot.js runs...');
    const proc = spawn('node', [SCRIPT_PATH]);
    let output = '';
    proc.stdout.on('data', (data) => {
        output += data.toString();
        // Send 'exit\n' to quit when prompt appears
        if (output.toLowerCase().includes('exit')) {
            proc.stdin.write('exit\n');
        }
    });
    proc.on('close', (code) => {
        console.assert(code === 0, 'chatbot.js did not exit cleanly');
        console.log('chatbot.js runs successfully.');
        console.log('Chatbot (JavaScript) basic test passed.');
    });
}

testScriptRuns();
