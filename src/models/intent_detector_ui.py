from __future__ import annotations

import html
import os

import gradio as gr

from intent_classifier import IntentClassifier


classifier = IntentClassifier()

INTENT_LABELS = {
    "greeting": "Greeting",
    "goodbye": "Goodbye",
    "gratitude": "Gratitude",
    "asking_mental_health_question": "Mental Health Question",
    "out_of_scope": "Out of Scope",
}

THEME = gr.themes.Base(
    primary_hue="amber",
    secondary_hue="cyan",
    neutral_hue="gray",
    radius_size="sm",
)

CSS = """
.intent-shell {
    max-width: 1040px;
    margin: 0 auto;
}
.intent-header {
    border-bottom: 3px solid #111827;
    padding: 18px 0 14px;
    margin-bottom: 18px;
}
.intent-kicker {
    color: #0891b2;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
}
.intent-panel {
    background: #ffffff;
    border: 2px solid #111827;
    box-shadow: 6px 6px 0 #facc15;
    padding: 16px;
}
.intent-result {
    background: #f9fafb;
    border: 2px solid #111827;
    padding: 16px;
}
.intent-label {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
}
.confidence-track {
    height: 12px;
    background: #e5e7eb;
    border: 1px solid #111827;
    margin-top: 12px;
}
.confidence-fill {
    height: 100%;
    background: #06b6d4;
}
.intent-note {
    color: #374151;
    margin-top: 10px;
}
.intent-error {
    background: #fff1f2;
    border: 2px solid #be123c;
    padding: 14px;
}
"""


def _result_card(intent: str, confidence: float, reason: str) -> str:
    label = INTENT_LABELS.get(intent, intent.replace("_", " ").title())
    safe_label = html.escape(label)
    safe_reason = html.escape(reason)
    width = max(0, min(confidence, 1)) * 100

    return (
        "<div class='intent-result'>"
        "<div>Detected intent</div>"
        f"<div class='intent-label'>{safe_label}</div>"
        f"<div class='confidence-track'><div class='confidence-fill' style='width: {width:.1f}%'></div></div>"
        f"<div class='intent-note'>Confidence: <b>{confidence:.1%}</b></div>"
        f"<div class='intent-note'>{safe_reason}</div>"
        "</div>"
    )


def _empty_result(message: str) -> tuple[str, list[list[str]]]:
    return f"<div class='intent-error'>{html.escape(message)}</div>", []


def predict_intent(text: str) -> tuple[str, list[list[str]]]:
    clean_text = (text or "").strip()
    if not clean_text:
        return _empty_result("Please enter a user message to classify.")

    try:
        result = classifier.classify(clean_text)
    except RuntimeError:
        return _empty_result("Groq API key is not configured. Set GROQ_API_KEY before running Module 3.")
    except ImportError:
        return _empty_result("Groq SDK is not installed. Install project requirements and try again.")
    except Exception as exc:
        print(f"Intent UI error: {type(exc).__name__}: {exc}")
        return _empty_result("Intent classification is unavailable right now. Please check the terminal logs.")

    intent = result["intent"]
    confidence = float(result["confidence"])
    reason = result["reason"]

    details = [
        ["Routing Key", intent],
        ["Display Label", INTENT_LABELS.get(intent, intent)],
        ["Confidence", f"{confidence:.1%}"],
        ["Reason", reason],
    ]
    return _result_card(intent, confidence, reason), details


with gr.Blocks(title="Intent Classifier") as interface:
    with gr.Column(elem_classes=["intent-shell"]):
        gr.HTML(
            """
            <div class="intent-header">
                <div class="intent-kicker">Module 3</div>
                <h1>Intent Routing</h1>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5, elem_classes=["intent-panel"]):
                text_input = gr.Textbox(
                    lines=7,
                    label="User message",
                    placeholder="Example: Hi, I feel anxious and cannot sleep.",
                )
                classify_button = gr.Button("Classify intent", variant="primary")
            with gr.Column(scale=4):
                result_output = gr.HTML()
                details_output = gr.Dataframe(
                    headers=["Field", "Value"],
                    datatype=["str", "str"],
                    label="Routing Details",
                    interactive=False,
                )

        classify_button.click(
            fn=predict_intent,
            inputs=text_input,
            outputs=[result_output, details_output],
        )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    interface.launch(theme=THEME, css=CSS, server_port=port)
