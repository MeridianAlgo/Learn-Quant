<<<<<<< HEAD
# Logging Utilities

This project provides simple logging utilities in both Python and JavaScript. You can add, read, edit, and delete log entries using either language. All logs are stored in a file named `log.txt` in the same directory.

---

## Python Logger

### Requirements
- Python 3.x (no external libraries required)

### Installation
No installation is required for Python. Just ensure you have Python 3 installed.

### Usage
1. Place `logger.py` in your project directory.
2. Run the logger with:
   ```sh
   python logger.py
   ```
3. You will see a menu:
   - Add log entry
   - Read all log entries
   - Edit a log entry
   - Delete a log entry
   - Exit
4. Follow the prompts to manage your log entries interactively.

---

## JavaScript Logger

### Requirements
- Node.js
- `readline-sync` package (for interactive CLI)

### Installation
1. Place `logger.js` in your project directory.
2. Install the required package (only needed once):
   ```sh
   npm install readline-sync
   ```

### Usage
1. Run the logger with:
   ```sh
   node logger.js
   ```
2. You will see a menu:
   - Add log entry
   - Read all log entries
   - Edit a log entry
   - Delete a log entry
   - Exit
3. Follow the prompts to manage your log entries interactively.

---

## Notes
- Both loggers use the same `log.txt` file by default. If you use both in the same directory, they will operate on the same log file.
- All functions are commented for clarity.

---

## License
This project is open source and free to use. 
=======
# Logging Utilities (NO API)

**This utility does NOT use any external APIs.** All logging is managed locally for learning and experimentation.

This project provides simple logging utilities in both Python and JavaScript. You can add, read, edit, and delete log entries using either language. All logs are stored in a file named `log.txt` in the same directory.

## Features
- Add, read, edit, and delete log entries
- Works in both Python and Node.js
- Interactive CLI menu
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.x (no external libraries required)
- Node.js (for JavaScript version)
- `readline-sync` package for Node.js:
  ```sh
  npm install readline-sync
  ```

## Setup
1. Clone or download this repository.
2. For Python:
   - Place `logger.py` in your project directory.
3. For Node.js:
   - Place `logger.js` in your project directory.
   - Install the required package:
     ```sh
     npm install readline-sync
     ```

## Usage Workflow (Step-by-Step)
1. Run the logger:
   - For Python:
     ```sh
     python logger.py
     ```
   - For Node.js:
     ```sh
     node logger.js
     ```
2. Follow the menu prompts to add, read, edit, or delete log entries.

**No real market data is used. This is for learning only!**

## Educational Notes
- **No API:** All logging is managed in Python or JavaScript, so you can see and modify the logic yourself.
- **How does it work?** The code is structured with functions and file I/O, with comments explaining each step.
- **How can you extend it?** Try adding timestamps, or exporting logs to CSV!

## License
MIT 
>>>>>>> 8944b09 (Initial commit: Comprehensive Python & JS Finance Utilities for Beginners (API & API-free, with detailed docs))
