# Gemini API Chatbot

<p class="lq-meta">Advanced <span>/</span> AI & Machine Learning <span>/</span> Python, JavaScript <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/AI%20Development">source</a></p>


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

## Running it

```bash
cd "AI Development"
python "chatbot.py"
node "chatbot.js"
```

## More in AI & Machine Learning

- [Learning Platform](Learning Platform.md): An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface.
- [Machine Learning - Cross Validation](Machine Learning - Cross Validation.md): A model scored on the data it was trained on is a student grading their own exam.
- [Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md): The dirty secret of quant machine learning: the model is rarely the bottleneck.
- [Machine Learning - Gradient Descent](Machine Learning - Gradient Descent.md): Gradient descent is the engine inside almost every model that learns.
- [Machine Learning - K Nearest Neighbors](Machine Learning - K Nearest Neighbors.md): Most models squeeze a training set down into a handful of parameters and then discard the data.
- [Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md): Given a few hundred stocks and their return characteristics, which ones behave alike?

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
