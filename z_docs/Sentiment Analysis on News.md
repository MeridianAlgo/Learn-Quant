<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">AI & Machine Learning</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Sentiment Analysis on News"
    python "sentiment_analysis.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Sentiment%20Analysis%20on%20News)

---
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


---

## Continue in AI & Machine Learning

<div class="grid cards" markdown>

-   :material-robot-outline: __[AI Development](AI Development.md)__

    Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js. This module demonstrates how to integrate a hosted large language model into a simple interactive application.

-   :material-robot-outline: __[Learning Platform](Learning Platform.md)__

    An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

-   :material-robot-outline: __[Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)__

    The dirty secret of quant machine learning: the model is rarely the bottleneck.

-   :material-robot-outline: __[Machine Learning - Gradient Descent](Machine Learning - Gradient Descent.md)__

    Gradient descent is the engine inside almost every model that learns. The idea

-   :material-robot-outline: __[Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md)__

    Given a few hundred stocks and their return characteristics, which ones behave

-   :material-robot-outline: __[Machine Learning - Logistic Regression](Machine Learning - Logistic Regression.md)__

    Linear regression predicts a number. **Logistic regression** predicts a

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
