from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    train = load_dataset("papluca/language-identification", split="train")
    valid = load_dataset("papluca/language-identification", split="validation")
    test  = load_dataset("papluca/language-identification", split="test")
    return train, valid, test


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])


def train(pipeline, train_dataset, valid_dataset):
    X_train = train_dataset["text"]
    y_train = train_dataset["labels"]
    X_valid = valid_dataset["text"]
    y_valid = valid_dataset["labels"]
    print(f"Training on {len(X_train)} samples across {len(set(y_train))} languages...")
    pipeline.fit(X_train, y_train)
    val_acc = accuracy_score(y_valid, pipeline.predict(X_valid))
    print(f"Validation accuracy: {val_acc:.4f}")
    return pipeline


def evaluate(pipeline, test_dataset):
    X_test = test_dataset["text"]
    y_test = test_dataset["labels"]
    preds  = pipeline.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nPer-language breakdown:")
    print(classification_report(y_test, preds))


def detect_language(pipeline, text: str) -> dict:
    proba    = pipeline.predict_proba([text])[0]
    classes  = pipeline.classes_
    scores   = dict(zip(classes, proba))
    best_lang = max(scores, key=scores.get)
    return {
        "language"  : best_lang,
        "confidence": round(scores[best_lang], 4),
        "scores"    : {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
    }


def save_model(pipeline, path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "language_detector.joblib")
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_model(path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "language_detector.joblib")
    return joblib.load(path)


if __name__ == "__main__":
    train_ds, valid_ds, test_ds = load_data()
    pipeline = build_pipeline()
    pipeline = train(pipeline, train_ds, valid_ds)
    evaluate(pipeline, test_ds)
    save_model(pipeline)
    samples = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "Guten Morgen, wie geht es Ihnen?",
        "مرحبا كيف حالك؟",
    ]
    for text in samples:
        result = detect_language(pipeline, text)
        print(f"  [{result['language']}] ({result['confidence']:.0%})  \"{text}\"")