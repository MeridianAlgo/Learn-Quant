<<<<<<< HEAD
# Gemini API Chatbot

This project provides simple command-line chatbots for Google's Gemini API in both Python and Node.js.

## Features
- Chat with Gemini using the latest high-throughput model (`gemini-2.5-flash`)
- Works in both Python and Node.js
- Easy to run from the terminal

## Requirements
- Node.js v18+ (for Node.js chatbot)
- Python 3.9+ (for Python chatbot)
- A valid Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup
1. **Clone or download this repository.**
2. **Install dependencies:**
   - For Python:
     ```sh
     pip install -r requirements.txt
     ```
   - For Node.js:
     ```sh
     npm install
     ```
3. **Set your Gemini API key:**
   - The API key is currently set directly in both `chatbot.py` and `chatbot.js` for demonstration.
   - **Recommended for production:** Use a `.env` file to store your API key securely.

## Using a `.env` File (Recommended)

1. **Create a file named `.env` in your project root:**
   ```
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
2. **Update your code:**
   - For Python, make sure you have these lines at the top:
     ```python
     from dotenv import load_dotenv
     load_dotenv()
     ```
     And use `os.getenv("GEMINI_API_KEY")` to access the key.
   - For Node.js, add this at the top:
     ```js
     import 'dotenv/config';
     ```
     And use `process.env.GEMINI_API_KEY` to access the key.
3. **Remove the hardcoded API key from your code for security.**

## Usage

### Python
Run the chatbot with:
```sh
python chatbot.py
```
Type your message and press Enter. Type `exit` to quit.

### Node.js
Run the chatbot with:
```sh
node chatbot.js
```
Type your message and press Enter. Type `exit` to quit.

## File Overview
- `chatbot.py`: Python CLI chatbot using the Gemini API.
- `chatbot.js`: Node.js CLI chatbot using the Gemini API.
- `requirements.txt`: Python dependencies.
- `package.json`: Node.js dependencies.
- `README.md`: This documentation file.

## Security Note
**Never share your API key publicly.** For production, always use environment variables or a `.env` file to keep your key secure.

## References
- [Gemini API Quickstart (Python)](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)
- [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits) 
=======
# Gemini API Chatbot (USES Gemini API)

**This utility uses the Gemini API for AI chat.** All other logic and data are managed locally for learning and experimentation.

This project provides simple command-line chatbots for Google's Gemini API in both Python and Node.js.

## Features
- Chat with Gemini using the latest high-throughput model (`gemini-2.5-flash`)
- Works in both Python and Node.js
- Easy to run from the terminal
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Node.js v18+ (for Node.js chatbot)
- Python 3.9+ (for Python chatbot)
- A valid Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup
1. Clone or download this repository.
2. Install dependencies:
   - For Python:
     ```sh
     pip install -r requirements.txt
     ```
   - For Node.js:
     ```sh
     npm install
     ```
3. Set your Gemini API key:
   - The API key is currently set directly in both `chatbot.py` and `chatbot.js` for demonstration.
   - **Recommended for production:** Use a `.env` file to store your API key securely.

## Usage Workflow (Step-by-Step)
1. Run the chatbot:
   - For Python:
     ```sh
     python chatbot.py
     ```
   - For Node.js:
     ```sh
     node chatbot.js
     ```
2. Type your message and press Enter. Type `exit` to quit.

## Educational Notes
- **How does it work?** The chatbot sends your message to the Gemini API and prints the response.
- **How is the code structured?** Each function is commented to explain its purpose. The code is designed for easy modification.
- **How can you extend it?** Try adding conversation history, or integrating with a GUI!

## Security Note
**Never share your API key publicly.** For production, always use environment variables or a `.env` file to keep your key secure.

## License
MIT

## References
- [Gemini API Quickstart (Python)](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)
- [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits) 
>>>>>>> 8944b09 (Initial commit: Comprehensive Python & JS Finance Utilities for Beginners (API & API-free, with detailed docs))
