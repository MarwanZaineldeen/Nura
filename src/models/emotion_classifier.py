from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "src" / "models" / "saved_emotion_model"
DEFAULT_HF_MODEL_ID = ""


def _load_transformer_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Module 2 requires torch and transformers. Install them with "
            "`python -m pip install -r requirements.txt`, or run the Colab notebook."
        ) from exc

    return torch, AutoModelForSequenceClassification, AutoTokenizer


class EmotionClassifier:
    """Transformer emotion classifier with confidence and simple word-occlusion explanations."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or os.getenv("EMOTION_MODEL_DIR", DEFAULT_MODEL_DIR))
        self.model_id = os.getenv("EMOTION_MODEL_ID", DEFAULT_HF_MODEL_ID).strip()
        self.active_model_source = str(self.model_dir if self.model_dir.exists() else self.model_id)
        self.torch = None
        self.tokenizer = None
        self.model = None
        self.id2label: dict[int, str] = {}

    def load_model(self) -> None:
        model_source = self._resolve_model_source()
        if not model_source:
            raise FileNotFoundError(
                "Emotion model is not available. Train Module 2 locally, or set "
                "EMOTION_MODEL_ID to a Hugging Face model repository."
            )

        torch, model_cls, tokenizer_cls = _load_transformer_stack()
        self.torch = torch
        self.tokenizer = tokenizer_cls.from_pretrained(model_source)
        self.model = model_cls.from_pretrained(model_source, low_cpu_mem_usage=True)
        self.model.eval()
        self.active_model_source = str(model_source)

        config_labels = self.model.config.id2label
        self.id2label = {int(key): value for key, value in config_labels.items()}

    def _resolve_model_source(self) -> str | Path | None:
        if self.model_dir.exists():
            return self.model_dir
        if self.model_id:
            return self.model_id
        return None

    def _score_text(self, text: str) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None or self.torch is None:
            self.load_model()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        inputs.pop("token_type_ids", None)

        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=-1)[0]

        best_index = int(probabilities.argmax().item())
        confidence = float(probabilities[best_index].item())
        scores = {self.id2label.get(index, str(index)): float(value.item()) for index, value in enumerate(probabilities)}

        return {
            "index": best_index,
            "emotion": self.id2label.get(best_index, str(best_index)),
            "confidence": confidence,
            "scores": scores,
        }

    def predict_with_confidence(self, text: str) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "is_confident": False,
                "message": "Please enter text to classify.",
            }

        prediction = self._score_text(clean_text)

        return {
            "emotion": prediction["emotion"],
            "confidence": prediction["confidence"],
            "is_confident": prediction["confidence"] >= 0.60,
            "message": None,
        }

    def explain(self, text: str, top_k: int = 8) -> dict[str, Any]:
        """Estimate influential words by measuring confidence drop after removing each word."""
        clean_text = text.strip()
        base_scores = self._score_text(clean_text)
        base_prediction = {
            "emotion": base_scores["emotion"],
            "confidence": base_scores["confidence"],
            "is_confident": base_scores["confidence"] >= 0.60,
            "message": None,
        }
        target_emotion = base_prediction["emotion"]
        base_confidence = base_prediction["confidence"]
        words = list(re.finditer(r"\b[\w']+\b", clean_text))

        impacts = []
        for match in words:
            reduced_text = (clean_text[: match.start()] + clean_text[match.end() :]).strip()
            reduced_scores = self._score_text(reduced_text) if reduced_text else {"scores": {target_emotion: 0.0}}
            target_confidence_without_word = reduced_scores["scores"].get(target_emotion, 0.0)
            confidence_drop = base_confidence - target_confidence_without_word

            if confidence_drop > 0.001:
                effect = "supports prediction"
            elif confidence_drop < -0.001:
                effect = "reduces prediction"
            else:
                effect = "neutral"

            impact = round(float(confidence_drop), 4)
            if impact == -0.0:
                impact = 0.0

            impacts.append(
                {
                    "word": match.group(0),
                    "impact": impact,
                    "confidence_without_word": round(float(target_confidence_without_word), 4),
                    "effect": effect,
                }
            )

        impacts = sorted(impacts, key=lambda item: item["impact"], reverse=True)
        return {
            "prediction": base_prediction,
            "top_evidence": impacts[:top_k],
            "all_evidence": impacts,
            "method": "word occlusion: larger impact means removing the word reduced confidence more",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module 2 emotion inference.")
    parser.add_argument("text", nargs="?", default="I feel anxious and overwhelmed today.")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classifier = EmotionClassifier(model_dir=args.model_dir)
    output = classifier.explain(args.text) if args.explain else classifier.predict_with_confidence(args.text)
    print(json.dumps(output, indent=2))
