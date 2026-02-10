# -----------------------------
# Sentiment Analysis on News Utility (NO API)
# -----------------------------
# This script lets you analyze the sentiment of news headlines using a simple rule-based approach.
# All analysis is done locally (no APIs, no ML models, no real market data).
# All code is commented for beginners to learn Python and basic NLP.
# -----------------------------

# Simple positive and negative word lists for rule-based sentiment
POSITIVE_WORDS = {
    "gain",
    "surge",
    "rise",
    "up",
    "beat",
    "strong",
    "profit",
    "growth",
    "record",
    "win",
    "positive",
    "bull",
    "optimistic",
    "soar",
    "improve",
}
NEGATIVE_WORDS = {
    "drop",
    "fall",
    "down",
    "miss",
    "weak",
    "loss",
    "decline",
    "cut",
    "negative",
    "bear",
    "pessimistic",
    "plunge",
    "worse",
    "fear",
}

# Store session results for summary
session_results = []


# Analyze a single headline for sentiment
def analyze_headline(headline):
    words = set(headline.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    score = pos - neg
    if score > 0:
        sentiment = "Positive"
    elif score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, score


# Print the main menu for the CLI
def print_menu():
    print("\nSentiment Analysis on News Menu:")
    print("1. Analyze headline")
    print("2. View session summary")
    print("3. Exit")


# Analyze a headline and store result
def analyze_and_store():
    headline = input("Enter headline: ")
    sentiment, score = analyze_headline(headline)
    print(f"Sentiment: {sentiment} (Score: {score})")
    session_results.append({"headline": headline, "sentiment": sentiment, "score": score})


# View summary of all analyzed headlines
def view_summary():
    if not session_results:
        print("No headlines analyzed yet.")
        return
    print("\nSession Summary:")
    for i, r in enumerate(session_results, 1):
        print(f"{i}. {r['headline']} => {r['sentiment']} (Score: {r['score']})")
    pos = sum(1 for r in session_results if r["sentiment"] == "Positive")
    neg = sum(1 for r in session_results if r["sentiment"] == "Negative")
    neu = sum(1 for r in session_results if r["sentiment"] == "Neutral")
    print(f"\nTotal: {len(session_results)} | Positive: {pos} | Negative: {neg} | Neutral: {neu}")


# Main workflow for the CLI
def main():
    print(
        """
====================================
Welcome to the Sentiment Analysis on News Utility!
This tool helps you learn Python and basic NLP by analyzing news headlines for sentiment.
- No APIs or ML models are used.
- All code is commented for beginners.
====================================
"""
    )
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            analyze_and_store()
        elif choice == "2":
            view_summary()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
