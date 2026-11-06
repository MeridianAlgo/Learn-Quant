# System Utilities

<p class="lq-meta">Beginner <span>/</span> Utilities & Tools <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/System%20Utilities">source</a></p>


This folder contains utilities for system-level operations, file management, and configuration in financial applications.

## Available Utilities

### File I/O (`file_io_utils.py`)
- JSON and CSV file operations
- Directory management
- File backup functionality
- File size formatting

### Configuration (`config_utils.py`)
- JSON configuration management
- Dot notation access to nested config values
- Configuration merging
- Environment variable integration

## Usage

```python
# File operations
from file_io_utils import read_json_file, write_json_file, backup_file
from config_utils import load_config, get_config_value, set_config_value

# Work with files
config = read_json_file("app_config.json")
backup_file("important_data.json")

# Configuration management
config = load_config("config.json")
api_key = get_config_value(config, "api.key")
set_config_value(config, "debug.mode", True)
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run each utility directly to see demonstrations:

```bash
python file_io_utils.py
python config_utils.py
```

## Common Use Cases

- **Configuration Management**: Handle application settings and preferences
- **Data Persistence**: Save and load trading data and portfolios
- **File Management**: Organize data files and backups
- **System Integration**: Work with external data sources
- **Application Setup**: Initialize and configure trading applications

---

## Running it

```bash
cd "System Utilities"
python "config_utils.py"
```

## More in Utilities & Tools

- [Core Utilities](Core Utilities.md): This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.
- [Currency Converter](Currency Converter.md): This tool lets you convert between currencies using user-supplied exchange rates.
- [Data Processing](Data Processing.md): This folder contains utilities for data processing, validation, and manipulation in financial applications.
- [Economic Calendar](Economic Calendar.md): This tool lets you simulate and explore economic events (like FOMC meetings, CPI releases, etc.) and see how they might impact markets.
- [Historical Data](Historical Data.md): A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API.
- [Logging](Logging.md): A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
