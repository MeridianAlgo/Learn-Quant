"""Command-line chatbot powered by Google's Gemini API.

The script is resilient to missing dependencies or API keys so it can be
executed in teaching environments without external configuration. If the
`google-genai` package is unavailable or the `GEMINI_API_KEY` environment
variable is not set to a real key, the script will exit gracefully.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def initialize_client() -> Optional["genai.Client"]:
    """Initialise the Gemini client when dependencies and keys are available."""

    try:
        from google import genai  # type: ignore import
    except ImportError:
        print("Gemini SDK not installed. Skipping interactive session.")
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("GEMINI_API_KEY not configured. Skipping interactive session.")
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to initialise Gemini client: {exc}")
        return None


def chat_loop(client: "genai.Client") -> None:
    """Interactive chat loop using an initialised Gemini client."""

    print("Gemini Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input,
            )
            output_text = getattr(response, "text", str(response))
            print("Gemini:", output_text)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Gemini response failed: {exc}")


def main() -> int:
    """Entry-point for the CLI script."""

    client = initialize_client()
    if client is None:
        return 0

    chat_loop(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())