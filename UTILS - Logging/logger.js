// logger.js
// Simple logging utility for adding, reading, editing, and deleting log entries in a text file.
// All log entries are stored in 'log.txt' in the same directory as this script.

const fs = require('fs');
const LOG_FILE = 'log.txt'; // Name of the log file

/**
 * Add a new log entry to the log file.
 * @param {string} entry - The string to be added as a log entry.
 */
function addLog(entry) {
    fs.appendFileSync(LOG_FILE, entry + '\n', 'utf8'); // Append entry with newline
}

/**
 * Read all log entries from the log file.
 * @returns {string[]} Array of log entry strings. Returns an empty array if file does not exist.
 */
function readLogs() {
    if (!fs.existsSync(LOG_FILE)) return []; // Return empty array if log file doesn't exist
    return fs.readFileSync(LOG_FILE, 'utf8').split('\n').filter(Boolean); // Split by newline and remove empty lines
}

/**
 * Edit a specific log entry by line number (0-based).
 * @param {number} lineNumber - Index of the log entry to edit (0-based).
 * @param {string} newEntry - New string to replace the old log entry.
 * @returns {boolean} True if edit was successful, False otherwise.
 */
function editLog(lineNumber, newEntry) {
    let logs = readLogs();
    if (lineNumber >= 0 && lineNumber < logs.length) {
        logs[lineNumber] = newEntry; // Replace the specified line
        fs.writeFileSync(LOG_FILE, logs.join('\n') + '\n', 'utf8'); // Write all lines back to the file
        return true;
    }
    return false; // Return false if lineNumber is out of range
}

/**
 * Delete a specific log entry by line number (0-based).
 * @param {number} lineNumber - Index of the log entry to delete (0-based).
 * @returns {boolean} True if deletion was successful, False otherwise.
 */
function deleteLog(lineNumber) {
    let logs = readLogs();
    if (lineNumber >= 0 && lineNumber < logs.length) {
        logs.splice(lineNumber, 1); // Remove the specified line
        fs.writeFileSync(LOG_FILE, logs.join('\n') + '\n', 'utf8'); // Write remaining lines back to the file
        return true;
    }
    return false; // Return false if lineNumber is out of range
}

// Example usage (uncomment to use):
// addLog('This is a new log entry');
// console.log(readLogs());
// editLog(0, 'This is an edited log entry');
// deleteLog(0);

// Export functions for use in other files (if needed)
module.exports = { addLog, readLogs, editLog, deleteLog };

if (require.main === module) {
    // Menu-driven CLI for logging operations
    const readlineSync = require('readline-sync');
    while (true) {
        console.log('\n--- Log Menu ---');
        console.log('1. Add log entry');
        console.log('2. Read all log entries');
        console.log('3. Edit a log entry');
        console.log('4. Delete a log entry');
        console.log('5. Exit');
        const choice = readlineSync.question('Choose an option (1-5): ');
        if (choice === '1') {
            const entry = readlineSync.question('Enter a log entry: ');
            addLog(entry);
            console.log('Log entry added.');
        } else if (choice === '2') {
            const logs = readLogs();
            if (logs.length) {
                console.log('\nAll log entries:');
                logs.forEach((log, i) => console.log(`${i}: ${log}`));
            } else {
                console.log('No log entries found.');
            }
        } else if (choice === '3') {
            const logs = readLogs();
            if (!logs.length) {
                console.log('No log entries to edit.');
                continue;
            }
            logs.forEach((log, i) => console.log(`${i}: ${log}`));
            const idx = parseInt(readlineSync.question('Enter the line number to edit: '), 10);
            if (isNaN(idx) || idx < 0 || idx >= logs.length) {
                console.log('Invalid line number.');
                continue;
            }
            const newEntry = readlineSync.question('Enter the new log entry: ');
            if (editLog(idx, newEntry)) {
                console.log('Log entry edited.');
            } else {
                console.log('Failed to edit log entry.');
            }
        } else if (choice === '4') {
            const logs = readLogs();
            if (!logs.length) {
                console.log('No log entries to delete.');
                continue;
            }
            logs.forEach((log, i) => console.log(`${i}: ${log}`));
            const idx = parseInt(readlineSync.question('Enter the line number to delete: '), 10);
            if (isNaN(idx) || idx < 0 || idx >= logs.length) {
                console.log('Invalid line number.');
                continue;
            }
            if (deleteLog(idx)) {
                console.log('Log entry deleted.');
            } else {
                console.log('Failed to delete log entry.');
            }
        } else if (choice === '5') {
            console.log('Exiting log program.');
            break;
        } else {
            console.log('Invalid choice. Please select 1-5.');
        }
    }
} 