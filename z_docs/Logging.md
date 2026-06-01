# Logging Utilities

A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

No external APIs or network access are used — logging is entirely local, making this module suitable for offline study of file I/O and CRUD patterns.

## Files

| File | Description |
|---|---|
| `logger.py` | Python implementation (standard library only) |
| `logger.js` | Node.js implementation (uses `readline-sync` for the CLI) |
| `log.txt` | Shared log store, created on first write |

## Requirements

- **Python**: 3.x — no third-party packages required.
- **Node.js**: any LTS release, plus the `readline-sync` package:
  ```sh
  npm install readline-sync
  ```

## Usage

**Python**

```sh
python logger.py
```

**Node.js**

```sh
node logger.js
```

Either entry point presents the same interactive menu:

- Add a log entry
- Read all log entries
- Edit a log entry
- Delete a log entry
- Exit

## Notes

- Both implementations default to the same `log.txt` file, so they can be used interchangeably within one directory and will operate on a shared log.
- Each function is documented inline to illustrate file handling and basic CRUD operations.

## License

MIT
