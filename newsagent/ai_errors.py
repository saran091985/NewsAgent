"""Turn OpenAI / network exceptions into one clear message for the screen."""

from __future__ import annotations


class AIError(RuntimeError):
    """The AI step could not run. The message is written for the user."""


def friendly(e: BaseException) -> str:
    text = str(e)
    name = type(e).__name__
    low = text.lower()
    if name == "AuthenticationError" or "401" in text or "incorrect api key" in low or "invalid_api_key" in low:
        return ("OpenAI rejected the API key (error 401: incorrect or revoked key). "
                "Put a valid key in OPENAI_API_KEY in the .env file in the NewsAgent folder, "
                "then restart the app.")
    if "insufficient_quota" in low or "exceeded your current quota" in low:
        return ("OpenAI says the account has no credit left (insufficient_quota). "
                "Add credit at platform.openai.com → Billing, then try again.")
    if name == "RateLimitError" or "429" in text:
        return "OpenAI is rate-limiting requests (error 429). Wait a minute and try again."
    if name in ("APIConnectionError", "ConnectError", "ConnectionError", "APITimeoutError") or "connection" in low:
        return "Could not reach OpenAI — check the internet connection and try again."
    if "api_key" in low and ("must be set" in low or "missing" in low or "not set" in low):
        return "No OpenAI key found. Add OPENAI_API_KEY=... to the .env file in the NewsAgent folder."
    return f"The AI call failed: {name}: {text[:300]}"
