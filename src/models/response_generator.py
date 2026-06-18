from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "llama-3.1-8b-instant"


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class ResponseGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        load_env_file()
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None

    def _get_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Set GROQ_API_KEY before generating chatbot responses.")

        if self.client is None:
            from groq import Groq

            self.client = Groq(api_key=self.api_key)

        return self.client

    def generate(self, state: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(state)},
        ]
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_completion_tokens=450,
            top_p=0.9,
        )
        content = completion.choices[0].message.content or "{}"
        return self._parse_response(content)

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", content.strip(), re.S)
        if match:
            content = match.group(0)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {
                "language_review": {"matches_module_1": None, "corrected_language_code": None, "reason": "Invalid JSON."},
                "intent_review": {"matches_module_3": None, "corrected_intent": None, "reason": "Invalid JSON."},
                "answer": content.strip(),
            }

        return {
            "language_review": parsed.get("language_review", {}),
            "intent_review": parsed.get("intent_review", {}),
            "answer": str(parsed.get("answer", "")).strip(),
        }

    @staticmethod
    def _system_prompt() -> str:
        return """You are a supportive mental-health chatbot.

Rules:
- Answer in the same language as the user.
- Recheck the language and intent using the user message, not only the earlier module outputs.
- Use the retrieved context as grounding, but do not copy long passages.
- Do not diagnose, prescribe medication, or claim to replace a professional.
- Be warm, practical, and concise.
- Make the user feel calmly supported and invited to continue.
- For non-crisis answers, end with one gentle, relevant follow-up question or grounding suggestion.
- If the message suggests immediate danger, self-harm, suicide, or harm to others, tell the user to contact local emergency services or the nearest emergency department immediately.
- If retrieved context is weak or unrelated, give a brief general supportive answer and suggest professional support.
- Return only valid JSON with keys: language_review, intent_review, answer.

JSON schema:
{
  "language_review": {
    "matches_module_1": true,
    "corrected_language_code": "en",
    "reason": "short explanation"
  },
  "intent_review": {
    "matches_module_3": true,
    "corrected_intent": "asking_mental_health_question",
    "reason": "short explanation"
  },
  "answer": "final user-facing answer"
}
"""

    @staticmethod
    def _user_prompt(state: dict[str, Any]) -> str:
        compact_state = {
            "user_message": state["user_message"],
            "language": state["language"],
            "emotion": state["emotion"],
            "intent": state["intent"],
            "retrieval": state["retrieval"],
            "conversation_memory": state.get("conversation_memory", []),
        }
        return (
            "Review the language and intent again, then answer the user.\n\n"
            "Pipeline state:\n"
            + json.dumps(compact_state, ensure_ascii=False, indent=2)
        )
