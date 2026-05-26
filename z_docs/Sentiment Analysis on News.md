# Sentiment Analysis on News Utility (NO API)

**This utility does NOT use any external APIs.** All sentiment analysis is done locally using a simple rule-based approach for learning and experimentation.

This tool lets you analyze the sentiment of news headlines or short texts using a basic positive/negative word list. You can enter headlines, see the sentiment score, and view a summary of results.

## Features
- Analyze sentiment of news headlines or short texts
- Uses a simple rule-based approach (positive/negative word lists)
- View sentiment score and summary (positive, negative, neutral)
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library)

## Setup
1. Copy `sentiment_analysis.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python sentiment_analysis.py
   ```
2. Follow the menu prompts:
   - Enter news headlines or short texts
   - View sentiment score and summary
   - Analyze multiple headlines in a session
   - Exit when done.

**No real market data or ML models are used. This is for learning only!**

## Example Session
```
Welcome to the Sentiment Analysis on News Utility!
1. Analyze headline
2. View session summary
3. Exit
Enter your choice: 1
Enter headline: Apple stock surges after strong earnings
Sentiment: Positive (Score: 2)
```

## Learning Notes
- **No API:** All analysis is managed in Python, so you can see and modify the logic yourself.
- **How does it work?** The code uses simple word lists to score sentiment, with comments explaining each step.
- **How can you extend it?** Try adding more words, or using a more advanced ML model!

## License
MIT
