<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">JavaScript</span></p>

!!! tip "Run this module"
    ```bash
    cd "News Fetching"
    node "fetchNews.js"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/News%20Fetching)

---
# Google News Fetcher

This utility provides a Google News headline scraper using the `google-news-json` package. It no longer requires any API keys, making it ideal for beginners who want to experiment with news-driven trading ideas or sentiment analysis without signing up for external services.

##  Quick Start

```bash
cd "UTILS - News Fetching"
npm install
node fetchNews.js
```

Follow the interactive prompts to:
- enter a company name or keyword (e.g., `AAPL`, `stock market`, `inflation`)
- choose a locale (default `en-US`)
- specify how many articles to retrieve (default `10`)

##  Features
- Keyword-driven Google News search
- Locale support (language-country format such as `en-US`, `fr-FR`)
- Configurable number of headlines to display
- Beginner-friendly comments and console guidance

##  Files
- `fetchNews.js` – interactive CLI script powered by `google-news-json`
- `package.json` – dependency list (`google-news-json` only)

##  Notes
- Results are scraped from Google News RSS feeds. Availability may vary by region and topic.
- Google may rate-limit excessive requests. Keep usage reasonable.
- Extend the script by saving article data to CSV/JSON, or by integrating with the sentiment analysis utility in `UTILS - Sentiment Analysis on News/`.

##  Related Learning Paths
- Beginner Python walkthroughs in `Documentation/Programs/level1_fundamentals.py`
- Sentiment analysis workflow in `UTILS - Sentiment Analysis on News/`

##  License
MIT

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
