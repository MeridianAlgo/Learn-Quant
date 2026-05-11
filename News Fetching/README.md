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