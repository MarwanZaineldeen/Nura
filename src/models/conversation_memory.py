from __future__ import annotations

import re


NAME_PATTERNS = [
    r"\bmy name is\s+([A-Z][a-zA-Z]{1,30})\b",
    r"\bcall me\s+([A-Z][a-zA-Z]{1,30})\b",
]

MEMORY_PATTERNS = [
    r"\bremember my name\b",
    r"\bwhat'?s my name\b",
    r"\bwhat is my name\b",
    r"\bdo you remember me\b",
    r"\bdid i tell you my name\b",
]


def extract_name(history: list[dict[str, str]]) -> str | None:
    for item in reversed(history):
        if item.get("role") != "user":
            continue

        text = item.get("content", "")
        for pattern in NAME_PATTERNS:
            match = re.search(pattern, text, flags=re.I)
            if match:
                return match.group(1)

    return None


def is_memory_question(message: str) -> bool:
    clean_message = message.lower()
    return any(re.search(pattern, clean_message) for pattern in MEMORY_PATTERNS)


def declared_name(message: str) -> str | None:
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, message, flags=re.I)
        if match:
            return match.group(1)
    return None


def memory_reply(message: str, history: list[dict[str, str]], language_code: str) -> str | None:
    name = declared_name(message)
    if name:
        return {
            "fr": f"Enchanté, {name}. Je m'en souviendrai pendant cette conversation. Qu'aimerais-tu explorer maintenant ?",
            "ar": f"تشرفت بمعرفتك يا {name}. سأتذكر اسمك خلال هذه المحادثة. ما الذي تحب أن نتحدث عنه الآن؟",
        }.get(
            language_code,
            f"Nice to meet you, {name}. I will remember your name during this conversation. What would you like to talk through next?",
        )

    if not is_memory_question(message):
        return None

    remembered_name = extract_name(history)
    if remembered_name:
        return {
            "fr": f"Oui, tu m'as dit que ton nom est {remembered_name}. Comment aimerais-tu que je t'aide maintenant ?",
            "ar": f"نعم، أخبرتني أن اسمك {remembered_name}. كيف يمكنني مساعدتك الآن؟",
        }.get(
            language_code,
            f"Yes, you told me your name is {remembered_name}. What would feel helpful to talk about now?",
        )

    return {
        "fr": "Je ne crois pas que tu m'aies donné ton nom dans cette conversation. Tu peux me le dire si tu veux.",
        "ar": "لا أعتقد أنك أخبرتني باسمك في هذه المحادثة. يمكنك أن تخبرني به إذا أردت.",
    }.get(
        language_code,
        "I do not think you have told me your name in this conversation yet. You can share it if you would like.",
    )
