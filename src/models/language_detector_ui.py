from pathlib import Path

import gradio as gr

from language_classifier import DEFAULT_MODEL_PATH, LanguageDetector


MODEL_PATH = Path(DEFAULT_MODEL_PATH)
detector = LanguageDetector(model_path=MODEL_PATH)
detector.load_model()


def predict_language(text: str) -> dict[str, str | float]:
    prediction = detector.predict_with_confidence(text or "")

    if prediction["message"]:
        return {
            "language": prediction["language_name"],
            "confidence": prediction["confidence"],
            "status": prediction["message"],
        }

    status = "Confident" if prediction["is_confident"] else "Uncertain"
    return {
        "language": prediction["language_name"],
        "confidence": round(prediction["confidence"], 4),
        "status": status,
    }


interface = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Type a question in any supported language...",
        label="User Question",
    ),
    outputs=gr.JSON(label="Detection Result"),
    title="Module 1: Language Detection",
    description="Character n-gram TF-IDF language classifier for routing chatbot queries.",
    flagging_mode="never",
)


if __name__ == "__main__":
    interface.launch()
