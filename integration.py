import os
import joblib
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"

from Module1 import detect_language, load_model as load_lang_model
from Module2 import load_model as load_emotion_model, detect_emotion, get_response_strategy
from Module3 import classify_intent
from Module4 import setup_rag_pipeline, rag_answer

LANG_CODE_TO_NAME = {
    "en": "English", "fr": "French", "ar": "Arabic",
    "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese", "tr": "Turkish",
}


def load_all_models():
    print("=" * 50)
    print("Loading all models...")
    print("[1/3] Loading language detector...")
    lang_pipeline = load_lang_model(os.path.join(BASE_DIR, "language_detector.joblib"))
    print("[2/3] Loading emotion model...")
    emotion_model, emotion_tokenizer = load_emotion_model()
    print("[3/3] Setting up RAG pipeline (Qdrant + BM25)...")
    bm25, all_chunks = setup_rag_pipeline(max_rows=2000)
    print("All models loaded.\n" + "=" * 50)
    return lang_pipeline, emotion_model, emotion_tokenizer, bm25, all_chunks


def translate_text(text: str, target_language: str) -> str:
    groq_client = Groq(api_key=GROQ_API_KEY)
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=300,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": f"Translate the following text to {target_language}. Return only the translation, no explanation:\n\n{text}"
        }]
    )
    return resp.choices[0].message.content.strip()


def process_message(
    user_message     : str,
    lang_pipeline,
    emotion_model,
    emotion_tokenizer,
    bm25,
    all_chunks,
    verbose          : bool = True,
) -> dict:
    print(f"\n{'='*50}")
    print(f"User: {user_message}")
    print(f"{'='*50}")

    lang_result   = detect_language(lang_pipeline, user_message)
    detected_lang = lang_result["language"]
    language_name = LANG_CODE_TO_NAME.get(detected_lang, detected_lang)
    if verbose:
        print(f"[Language] {detected_lang} → {language_name} ({lang_result['confidence']:.0%})")

    intent_result = classify_intent(user_message)
    intent        = intent_result["intent"]
    confidence    = intent_result["confidence"]
    if verbose:
        print(f"[Intent]   {intent} ({confidence:.0%}) — {intent_result['reason']}")

    simple_responses = {
        "greeting"    : "Hello! I'm here to support you. How are you feeling today?",
        "goodbye"     : "Take care of yourself. Remember, it's okay to reach out whenever you need support. Goodbye!",
        "gratitude"   : "You're very welcome. I'm always here if you need to talk.",
        "out_of_scope": "I'm a mental health support assistant. I'm best equipped to help with emotional wellbeing questions. Could you tell me how you're feeling?",
    }

    if intent in simple_responses:
        response_text = simple_responses[intent]
        if detected_lang != "en":
            if verbose:
                print(f"[Translate] Translating response to {language_name}...")
            response_text = translate_text(response_text, language_name)
        return _build_response(response_text, detected_lang, intent, emotion=None, sources=[])

    emotion_result = detect_emotion(emotion_model, emotion_tokenizer, user_message)
    strategy       = get_response_strategy(emotion_result["emotion"])
    if verbose:
        print(f"[Emotion]  {emotion_result['emotion']} ({emotion_result['confidence']:.0%})")
        print(f"[Strategy] {strategy}")

    rag_result = rag_answer(
        user_question     = user_message,
        bm25              = bm25,
        all_chunks        = all_chunks,
        emotion_result    = emotion_result,
        emotion_strategy  = strategy,
        response_language = language_name,
        verbose           = verbose,
    )

    return _build_response(
        response = rag_result["answer"],
        language = detected_lang,
        intent   = intent,
        emotion  = emotion_result,
        sources  = rag_result["sources"],
    )


def _build_response(response, language, intent, emotion, sources) -> dict:
    return {
        "response": response,
        "language": language,
        "intent"  : intent,
        "emotion" : emotion,
        "sources" : sources,
    }


if __name__ == "__main__":
    lang_pipeline, emotion_model, emotion_tokenizer, bm25, all_chunks = load_all_models()
    test_messages = [
        "Hey there!",
        "I've been having panic attacks and don't know what to do.",
        "How do I cope with anxiety at work?",
        "Bonjour, comment allez-vous?",
        "كيف أتعامل مع القلق والتوتر؟",
        "Thank you so much, this really helped.",
        "What's the best programming language to learn?",
        "I feel completely hopeless lately.",
        "Goodbye, thanks for everything.",
    ]
    for msg in test_messages:
        result = process_message(
            msg, lang_pipeline, emotion_model, emotion_tokenizer, bm25, all_chunks, verbose=True,
        )
        print(f"\nBot: {result['response']}")
        if result["sources"]:
            print(f"Sources used: {len(result['sources'])}")
        print("-" * 50)