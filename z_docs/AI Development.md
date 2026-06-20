<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">AI & Machine Learning</span><span class="lq-badge lq-lang">Python · JavaScript</span></p>

!!! tip "Run this module"
    ```bash
    cd "AI Development"
    python "chatbot.py"
    node "chatbot.js"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/AI%20Development)

---
# Gemini API Chatbot

Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js. This module demonstrates how to integrate a hosted large language model into a simple interactive application.

> **External API:** This utility calls the Gemini API for chat responses. A network connection and a valid API key are required. All surrounding logic runs locally.

## Files

| File | Description |
|---|---|
| `chatbot.py` | Python CLI chatbot using the Gemini API |
| `chatbot.js` | Node.js CLI chatbot using the Gemini API |
| `requirements.txt` | Python dependencies |
| `package.json` | Node.js dependencies |

## Requirements

- **Python**: 3.9+
- **Node.js**: v18+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

The default model is `gemini-2.5-flash`, chosen for low latency and high throughput.

## Setup

1. Install dependencies:
   ```sh
   pip install -r requirements.txt   # Python
   npm install                       # Node.js
   ```
2. Provide your API key via a `.env` file (recommended over hardcoding):
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   - **Python** — load it at startup:
     ```python
     from dotenv import load_dotenv
     load_dotenv()
     # access with os.getenv("GEMINI_API_KEY")
     ```
   - **Node.js** — load it at startup:
     ```js
     import 'dotenv/config';
     // access with process.env.GEMINI_API_KEY
     ```

## Usage

```sh
python chatbot.py   # Python
node chatbot.js     # Node.js
```

Type a message and press Enter to chat. Type `exit` to quit.

## Security

Never commit or share your API key. Always load credentials from environment variables or a `.env` file, and keep that file out of version control.

## References

- [Gemini API Quickstart (Python)](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)
- [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)

## License

MIT


---

## Continue in AI & Machine Learning

<div class="grid cards" markdown>

-   :material-robot-outline: __[Learning Platform](Learning Platform.md)__

    An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

-   :material-robot-outline: __[Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)__

    The dirty secret of quant machine learning: the model is rarely the bottleneck.

-   :material-robot-outline: __[Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md)__

    Given a few hundred stocks and their return characteristics, which ones behave

-   :material-robot-outline: __[Machine Learning - Logistic Regression](Machine Learning - Logistic Regression.md)__

    Linear regression predicts a number. **Logistic regression** predicts a

-   :material-robot-outline: __[Machine Learning - Random Forest](Machine Learning - Random Forest.md)__

    This module provides a basic implementation of a Random Forest Predictor for quantitative finance. It uses scikit-learn's `RandomForestRegressor` to predict time series data or returns based on a set of features.

-   :material-robot-outline: __[Machine Learning Time Series](Machine Learning Time Series.md)__

    Applying incredibly sophisticated statistical and advanced computational matrix calculating algorithms to historical sequential asset prices explicitly enables quantitative researchers to discover heavily latent non linear correlation patterns. Standard basic linear techniques lack the internal theoretical mapping memory required to fully process continuous progression data natively. Therefore, explicit sequential data pattern prediction necessitates deeply specialized memory architectures uniquely capable of successfully retaining vast contextual numerical memory safely across thousands of chronologically independent market observations simultaneously.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
