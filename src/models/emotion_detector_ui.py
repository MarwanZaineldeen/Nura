from pathlib import Path
import os

import gradio as gr

from emotion_classifier import DEFAULT_MODEL_DIR, EmotionClassifier


classifier = EmotionClassifier()

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="rose",
    neutral_hue="zinc",
    radius_size="sm",
)

CSS = """
.emotion-shell {
    max-width: 980px;
    margin: 0 auto;
}
.status-box {
    border-left: 4px solid #0f766e;
    padding: 12px 14px;
    background: #f8fafc;
}
.missing-box {
    border-left: 4px solid #be123c;
    padding: 12px 14px;
    background: #fff1f2;
}
.emotion-card {
    border: 1px solid #d4d4d8;
    padding: 16px;
    background: white;
}
.emotion-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f766e;
}
"""


def model_status() -> str:
    model_dir = Path(os.getenv("EMOTION_MODEL_DIR", DEFAULT_MODEL_DIR))
    if model_dir.exists():
        return f"<div class='status-box'>Using local trained model: <code>{model_dir}</code></div>"
    return (
        "<div class='missing-box'>Local emotion model is not available at "
        f"<code>{model_dir}</code>. Copy the trained "
        "<code>saved_emotion_model</code> folder there, or set "
        "<code>EMOTION_MODEL_DIR</code> to its current location.</div>"
    )


def _empty_result(message: str) -> tuple[str, list[list[str | float]], str]:
    return (
        f"<div class='missing-box'>{message}</div>",
        [],
        model_status(),
    )


def predict_emotion(text: str) -> tuple[str, list[list[str | float]], str]:
    if not (text or "").strip():
        return _empty_result("Please enter a message to analyze.")

    try:
        result = classifier.explain(text or "", top_k=6)
        emotion = result["prediction"]["emotion"]
        confidence = result["prediction"]["confidence"]
        card = (
            "<div class='emotion-card'>"
            "<div>Predicted emotion</div>"
            f"<div class='emotion-value'>{emotion.title()}</div>"
            f"<div>Confidence: <b>{confidence:.1%}</b></div>"
            "</div>"
        )
        evidence = [[item["word"], item["impact"]] for item in result["top_evidence"]]
        status = f"<div class='status-box'>Model source: <code>{classifier.active_model_source}</code></div>"
        return card, evidence, status
    except FileNotFoundError:
        return _empty_result("Local emotion model is not available yet.")
    except ImportError as exc:
        return _empty_result(f"Missing dependency: {exc}")
    except Exception as exc:
        return _empty_result(f"Emotion analysis is unavailable right now: {exc}")


with gr.Blocks(title="Emotion Classifier") as interface:
    with gr.Column(elem_classes=["emotion-shell"]):
        gr.Markdown(
            """
            # Emotion Classification
            DistilBERT-based emotion analysis with confidence and word-level evidence.
            """
        )
        status = gr.HTML(value=model_status())

        with gr.Row():
            with gr.Column(scale=5):
                text_input = gr.Textbox(
                    lines=7,
                    label="User message",
                    placeholder="Example: I feel overwhelmed and I cannot sleep.",
                )
                analyze_button = gr.Button("Analyze emotion", variant="primary")
            with gr.Column(scale=4):
                result_output = gr.HTML(label="Prediction")
                evidence_output = gr.Dataframe(
                    headers=["Word", "Impact"],
                    datatype=["str", "number"],
                    label="Word Evidence",
                    interactive=False,
                )
                summary_output = gr.HTML()

        analyze_button.click(
            fn=predict_emotion,
            inputs=text_input,
            outputs=[result_output, evidence_output, summary_output],
        )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    interface.launch(theme=THEME, css=CSS, server_port=port)
