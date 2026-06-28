from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 12.0


class GoogleAIStudioClient:
    """Small REST client for Google AI Studio Gemini fallback calls."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GOOGLE_GENERATION_MODEL", DEFAULT_GOOGLE_MODEL)
        self.timeout_seconds = timeout_seconds or float(os.getenv("GOOGLE_REQUEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY to use Google AI Studio fallback.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.9,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:240]
            raise RuntimeError(f"Google AI Studio HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Google AI Studio request failed: {error.reason}") from error

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Google AI Studio returned an unexpected response shape.") from error
