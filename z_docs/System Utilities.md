<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "System Utilities"
    python "config_utils.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/System%20Utilities)

---
# System Utilities

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

## Continue in Utilities & Tools

<div class="grid cards" markdown>

-   :material-tools: __[Core Utilities](Core Utilities.md)__

    This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

-   :material-tools: __[Currency Converter](Currency Converter.md)__

    **This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

-   :material-tools: __[Data Processing](Data Processing.md)__

    This folder contains utilities for data processing, validation, and manipulation in financial applications.

-   :material-tools: __[Economic Calendar](Economic Calendar.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-tools: __[Historical Data](Historical Data.md)__

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   :material-tools: __[Logging](Logging.md)__

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
