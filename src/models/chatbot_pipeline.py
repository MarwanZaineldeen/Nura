from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
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
from response_generator import ResponseGenerator
from safety_router import crisis_reply, detect_crisis
from src.retrieval.retrieval_engine import RetrievalEngine


class ChatbotPipeline:
    def __init__(self, retrieval_source: str = "both", top_k: int = 5, retrieval_collection: str | None = None) -> None:
        self.retrieval_source = retrieval_source
        self.top_k = top_k
        self.retrieval_collection = retrieval_collection
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

        conversation_history = self._prepare_history(clean_message, history or [])
        state = self._analyze(clean_message, conversation_history)
        state["conversation_history"] = conversation_history
        language_code = state["language"].get("language_code", "en")

        if state["safety"]["is_crisis"]:
            state["route"] = "crisis"
            return {"response": crisis_reply(language_code), "state": state}

        intent = state["intent"]["intent"]
        use_retrieval = intent == "asking_mental_health_question"
        state["route"] = "rag" if use_retrieval else "direct_response"
        state["retrieval"] = {
            "enabled": use_retrieval,
            "source": self.retrieval_source,
            "top_k": self.top_k,
            "results": [],
        }
        if use_retrieval:
            retrieval_query = state["intent"].get("retrieval_query") or clean_message
            state["retrieval"]["query"] = retrieval_query
            try:
                state["retrieval"]["results"] = self._retrieve(retrieval_query)
            except Exception as error:
                state["retrieval"]["error"] = f"{type(error).__name__}"

        try:
            generated = self.response_generator.generate(state)
            state["llm_review"] = {
                "language": generated.get("language_review", {}),
                "emotion": generated.get("emotion_review", {}),
                "intent": generated.get("intent_review", {}),
            }
            corrected_intent = state["llm_review"]["intent"].get("corrected_intent")
            state["final_intent"] = corrected_intent or state["intent"].get("intent")
            state["final_route"] = "rag" if state["final_intent"] == "asking_mental_health_question" else "direct_response"
            if state["final_intent"] == "asking_mental_health_question":
                state["suggested_questions"] = generated.get("suggested_questions", [])
            else:
                state["suggested_questions"] = []
            response = generated.get("answer") or "I am here with you, but I could not generate a complete response."
        except RuntimeError as error:
            state["llm_review"] = {"language": {}, "emotion": {}, "intent": {}}
            state["suggested_questions"] = []
            state["generation_error"] = str(error)
            response = (
                "I'm here with you \u2764\ufe0f. I can listen, help you slow things down, and support you with mental-health questions. "
                "The advanced response generator is not available right now, so please try again shortly. "
                "If this feels urgent or unsafe, contact local emergency support or someone you trust right away."
            )
        except Exception as error:
            state["llm_review"] = {"language": {}, "emotion": {}, "intent": {}}
            state["suggested_questions"] = []
            state["generation_error"] = f"{type(error).__name__}: {error}"
            response = (
                "I am here with you, but I could not complete a full answer at the moment. "
                "Try again shortly, or contact a trusted person or professional support if you need help now."
            )

        return {"response": response, "suggested_questions": state.get("suggested_questions", []), "state": state}

    def _analyze(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        safety = detect_crisis(message)

        with ThreadPoolExecutor(max_workers=3) as executor:
            language_future = executor.submit(self._safe_language, message)
            emotion_future = executor.submit(self._safe_emotion, message)
            if safety["is_crisis"]:
                intent_future = None
            else:
                intent_future = executor.submit(self._safe_intent, message, history)

            language = language_future.result()
            emotion = emotion_future.result()
            intent = self._crisis_intent(message) if intent_future is None else intent_future.result()

        return {
            "user_message": message,
            "language": language,
            "emotion": emotion,
            "intent": intent,
            "safety": safety,
            "retrieval": {"enabled": False, "results": []},
        }

    def _crisis_intent(self, message: str) -> dict[str, Any]:
        return {
            "intent": "asking_mental_health_question",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "intent_scores": {},
            "reason": "Crisis guardrail matched before live intent classification.",
            "retrieval_query": message,
            "contextual_follow_up": False,
            "interaction_type": "standalone",
            "classification_skipped": True,
        }

    def _safe_intent(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        try:
            return self.intent_classifier.classify(message, history=history)
        except Exception as error:
            return self._fallback_intent(message, error)

    def set_retrieval_collection(self, collection_name: str | None) -> None:
        if collection_name != self.retrieval_collection:
            self.retrieval_collection = collection_name
            self.retrieval_engine = None

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
        return {
            "intent": "out_of_scope",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "intent_scores": {},
            "reason": f"Intent classification unavailable: {type(error).__name__}.",
            "retrieval_query": message,
            "contextual_follow_up": False,
            "interaction_type": "standalone",
        }

    @staticmethod
    def _prepare_history(message: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
        clean_history = [
            {"role": item.get("role", ""), "content": str(item.get("content", "")).strip()}
            for item in history
            if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
        ]
        if clean_history and clean_history[-1]["role"] == "user" and clean_history[-1]["content"] == message:
            clean_history.pop()
        return clean_history[-8:]

    def _retrieve(self, message: str) -> list[dict[str, Any]]:
        if self.retrieval_engine is None:
            self.retrieval_engine = RetrievalEngine(collection_name=self.retrieval_collection)
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
