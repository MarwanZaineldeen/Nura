from flask import Flask, request, jsonify
from integration import load_all_models, process_message

app    = Flask(__name__)
models = {}


def initialize():
    lang_pipeline, emotion_model, emotion_tokenizer, bm25, all_chunks = load_all_models()
    models["lang_pipeline"]     = lang_pipeline
    models["emotion_model"]     = emotion_model
    models["emotion_tokenizer"] = emotion_tokenizer
    models["bm25"]              = bm25
    models["all_chunks"]        = all_chunks


@app.route("/")
def root():
    return jsonify({"message": "Mental Health Support API is running. POST to /chat to start."})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": list(models.keys())})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Message cannot be empty."}), 400
    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    result = process_message(
        user_message      = user_message,
        lang_pipeline     = models["lang_pipeline"],
        emotion_model     = models["emotion_model"],
        emotion_tokenizer = models["emotion_tokenizer"],
        bm25              = models["bm25"],
        all_chunks        = models["all_chunks"],
        verbose           = True,
    )
    return jsonify({
        "response": result["response"],
        "language": result["language"],
        "intent"  : result["intent"],
        "emotion" : result["emotion"],
        "sources" : result["sources"],
    })


if __name__ == "__main__":
    initialize()
    app.run(debug=False, host="0.0.0.0", port=7860)