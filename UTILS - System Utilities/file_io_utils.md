# File I/O Utilities

This module provides comprehensive file input/output utilities for financial applications, including JSON and CSV operations, directory management, file backup functionality, and file size formatting.

## Functions

### `ensure_directory_exists(file_path: str) -> None`
Ensures that the directory for the given file path exists.

**Parameters:**
- `file_path`: File path that requires directory existence

**Example:**
```python
>>> ensure_directory_exists("data/trades/2024/01/trades.json")
# Creates directory structure if it doesn't exist
```

### `read_json_file(file_path: str) -> Dict[str, Any]`
Reads and parses JSON file.

**Parameters:**
- `file_path`: Path to JSON file

**Returns:**
- Parsed JSON data as dictionary

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `json.JSONDecodeError`: If file contains invalid JSON

**Example:**
```python
>>> data = read_json_file("config.json")
>>> print(data["api_key"])
"your_api_key_here"
```

### `write_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> None`
Writes data to JSON file.

**Parameters:**
- `data`: Data to write
- `file_path`: Output file path
- `indent`: JSON indentation level

**Example:**
```python
>>> portfolio_data = {"AAPL": 100, "GOOGL": 50}
>>> write_json_file(portfolio_data, "portfolio.json")
```

### `read_csv_file(file_path: str, delimiter: str = ',') -> List[Dict[str, str]]`
Reads CSV file and returns as list of dictionaries.

**Parameters:**
- `file_path`: Path to CSV file
- `delimiter`: CSV delimiter character

**Returns:**
- List of rows as dictionaries

**Example:**
```python
>>> trades = read_csv_file("trades.csv")
>>> print(trades[0]["symbol"])
"AAPL"
```

### `write_csv_file(data: List[Dict[str, str]], file_path: str, delimiter: str = ',') -> None`
Writes data to CSV file.

**Parameters:**
- `data`: List of dictionaries to write
- `file_path`: Output file path
- `delimiter`: CSV delimiter character

**Example:**
```python
>>> trades = [{"symbol": "AAPL", "shares": 100, "price": 150.25}]
>>> write_csv_file(trades, "trades.csv")
```

### `backup_file(file_path: str, backup_suffix: str = '.bak') -> str`
Creates a backup of the specified file.

**Parameters:**
- `file_path`: Path to file to backup
- `backup_suffix`: Suffix for backup file

**Returns:**
- Path to backup file

**Example:**
```python
>>> backup_path = backup_file("important_data.json")
>>> print(backup_path)
"important_data.json.bak"
```

### `get_file_size(file_path: str) -> int`
Gets file size in bytes.

**Parameters:**
- `file_path`: Path to file

**Returns:**
- File size in bytes

**Example:**
```python
>>> size = get_file_size("large_file.csv")
>>> print(size)
1048576
```

### `format_file_size(size_bytes: int) -> str`
Formats file size in human-readable format.

**Parameters:**
- `size_bytes`: Size in bytes

**Returns:**
- Human-readable size string

**Example:**
```python
>>> format_file_size(1048576)
"1.0 MB"
>>> format_file_size(1536)
"1.5 KB"
```

## Usage

```python
from file_io_utils import (
    read_json_file, write_json_file, read_csv_file, write_csv_file,
    backup_file, ensure_directory_exists, format_file_size
)

# Work with JSON configuration
config = read_json_file("config.json")
config["new_setting"] = "value"
write_json_file(config, "config.json")

# Backup important file before modification
backup_file("portfolio.json")

# Work with CSV trade data
trades = read_csv_file("trades.csv")
new_trades = [{"symbol": "TSLA", "shares": 10, "price": 800.50}]
write_csv_file(new_trades, "new_trades.csv")

# Ensure directory exists for data files
ensure_directory_exists("data/2024/01/trades.json")
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run the module directly to see demonstrations:

```bash
python file_io_utils.py
```

## Common Use Cases

- **Configuration Management**: Read and write JSON configuration files
- **Data Export**: Save trading data to CSV format
- **Data Import**: Load historical data from CSV files
- **Backup Operations**: Create automatic backups of important files
- **File Management**: Organize data files in directory structures
- **Data Persistence**: Save application state and user preferences
