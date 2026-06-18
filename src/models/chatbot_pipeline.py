from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from emotion_classifier import EmotionClassifier
from intent_classifier import IntentClassifier
from language_classifier import LanguageDetector
from conversation_memory import memory_reply
from response_generator import ResponseGenerator
from safety_router import crisis_reply, detect_crisis, simple_reply
from src.retrieval.retrieval_engine import RetrievalEngine


class ChatbotPipeline:
    def __init__(self, retrieval_source: str = "both", top_k: int = 5) -> None:
        self.retrieval_source = retrieval_source
        self.top_k = top_k
        self.language_detector = LanguageDetector()
        self.language_detector.load_model()
        self.emotion_classifier = EmotionClassifier()
        self.intent_classifier = IntentClassifier()
        self.retrieval_engine = None
        self.response_generator = ResponseGenerator()

    def run(self, user_message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        clean_message = user_message.strip()
        if not clean_message:
            return {"response": "Please enter a message.", "state": {}}

        conversation_memory = (history or [])[-8:]
        state = self._analyze(clean_message, conversation_memory)
        state["conversation_memory"] = conversation_memory
        language_code = state["language"].get("language_code", "en")

        if state["safety"]["is_crisis"]:
            state["route"] = "crisis"
            return {"response": crisis_reply(language_code), "state": state}

        memory_response = memory_reply(clean_message, conversation_memory, language_code)
        if memory_response:
            state["route"] = "memory"
            return {"response": memory_response, "state": state}

        intent = state["intent"]["intent"]
        if intent != "asking_mental_health_question":
            state["route"] = "simple_reply"
            return {"response": simple_reply(intent, language_code), "state": state}

        state["route"] = "rag"
        state["retrieval"] = {
            "enabled": True,
            "source": self.retrieval_source,
            "top_k": self.top_k,
            "results": [],
        }
        try:
            state["retrieval"]["results"] = self._retrieve(clean_message)
        except Exception as error:
            state["retrieval"]["error"] = f"{type(error).__name__}"

        try:
            generated = self.response_generator.generate(state)
            state["llm_review"] = {
                "language": generated.get("language_review", {}),
                "intent": generated.get("intent_review", {}),
            }
            response = generated.get("answer") or "I am here with you, but I could not generate a complete response."
        except RuntimeError as error:
            state["llm_review"] = {"language": {}, "intent": {}}
            state["generation_error"] = str(error)
            response = (
                "I am here with you, but response generation is not available right now. "
                "If this is urgent or you feel unsafe, please contact local emergency support."
            )
        except Exception as error:
            state["llm_review"] = {"language": {}, "intent": {}}
            state["generation_error"] = f"{type(error).__name__}: {error}"
            response = (
                "I am here with you, but I could not complete a full answer at the moment. "
                "Try again shortly, or contact a trusted person or professional support if you need help now."
            )

        return {"response": response, "state": state}

    def _analyze(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        safety = detect_crisis(message)
        memory_response = memory_reply(message, history, "en")

        if safety["is_crisis"]:
            intent = {
                "intent": "asking_mental_health_question",
                "confidence": 1.0,
                "reason": "Crisis guardrail matched before live intent classification.",
            }
        elif memory_response:
            intent = {
                "intent": "out_of_scope",
                "confidence": 1.0,
                "reason": "Conversation-memory follow-up handled without RAG.",
            }
        else:
            try:
                intent = self.intent_classifier.classify(message)
            except Exception as error:
                intent = self._fallback_intent(message, error)

        return {
            "user_message": message,
            "language": self._safe_language(message),
            "emotion": self._safe_emotion(message),
            "intent": intent,
            "safety": safety,
            "retrieval": {"enabled": False, "results": []},
        }

    def _safe_language(self, message: str) -> dict[str, Any]:
        try:
            return self.language_detector.predict_with_confidence(message)
        except Exception as error:
            return {
                "language_code": "en",
                "language_name": "English",
                "confidence": 0.0,
                "is_confident": False,
                "message": f"Language detection unavailable: {type(error).__name__}",
            }

    def _safe_emotion(self, message: str) -> dict[str, Any]:
        try:
            return self.emotion_classifier.predict_with_confidence(message)
        except Exception as error:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "is_confident": False,
                "scores": {},
                "message": f"Emotion detection unavailable: {type(error).__name__}",
            }

    def _fallback_intent(self, message: str, error: Exception) -> dict[str, Any]:
        clean = message.lower().strip()
        if clean in {"hi", "hello", "hey", "good morning", "good evening"}:
            intent = "greeting"
        elif clean in {"thanks", "thank you", "thx"}:
            intent = "gratitude"
        elif clean in {"bye", "goodbye", "see you", "see you later"}:
            intent = "goodbye"
        else:
            intent = "out_of_scope"

        return {
            "intent": intent,
            "confidence": 0.0,
            "reason": f"Intent classification unavailable; fallback used: {type(error).__name__}.",
        }

    def _retrieve(self, message: str) -> list[dict[str, Any]]:
        if self.retrieval_engine is None:
            self.retrieval_engine = RetrievalEngine()
        return self.retrieval_engine.search(message, source=self.retrieval_source, top_k=self.top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated mental-health chatbot pipeline.")
    parser.add_argument("message", nargs="?", default="I feel anxious and cannot sleep.")
    parser.add_argument("--source", choices=["both", "cci", "amod"], default="both")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = ChatbotPipeline(retrieval_source=args.source, top_k=args.top_k)
    output = pipeline.run(args.message)
    print(json.dumps(output, indent=2, ensure_ascii=False))
