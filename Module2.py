import os
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "emotion_model")

MODEL_NAME    = "distilbert-base-uncased"
MAX_LENGTH    = 128
BATCH_SIZE    = 32
EPOCHS        = 3
LEARNING_RATE = 2e-5
LABELS   = ["sadness", "joy", "love", "anger", "fear", "surprise"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_data():
    dataset = load_dataset("dair-ai/emotion")
    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    return dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def train():
    dataset   = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = tokenize_dataset(dataset, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print(f"\nFine-tuning {MODEL_NAME} on dair-ai/emotion...")
    trainer.train()
    print("\n── Test Set Evaluation ──")
    preds_output = trainer.predict(tokenized["test"])
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = np.array(tokenized["test"]["labels"])
    print(f"Test Accuracy: {accuracy_score(labels, preds):.4f}\n")
    print(classification_report(labels, preds, target_names=LABELS))
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\nModel saved to {MODEL_DIR}/")
    return model, tokenizer


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def detect_emotion(model, tokenizer, text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs        = torch.softmax(logits, dim=-1).squeeze().tolist()
    scores       = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}
    best_emotion = max(scores, key=scores.get)
    return {
        "emotion"   : best_emotion,
        "confidence": round(scores[best_emotion], 4),
        "scores"    : dict(sorted(scores.items(), key=lambda x: -x[1])),
    }


EMOTION_STRATEGY = {
    "sadness" : "Respond with empathy and emotional support. Acknowledge their feelings first.",
    "joy"     : "Match their energy with an upbeat, enthusiastic tone.",
    "love"    : "Respond warmly and positively, mirror their affection.",
    "anger"   : "Stay calm, validate frustration, de-escalate before solving.",
    "fear"    : "Be reassuring and clear. Avoid uncertainty in your response.",
    "surprise": "Acknowledge the unexpected element, then provide clarity.",
}

def get_response_strategy(emotion: str) -> str:
    return EMOTION_STRATEGY.get(emotion, "Respond in a neutral, helpful tone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true")
    args, unknown = parser.parse_known_args()
    if args.infer:
        model, tokenizer = load_model()
    else:
        model, tokenizer = train()
    samples = [
        "I just got promoted, I'm so happy!",
        "I feel so hopeless and empty inside.",
        "I'm absolutely furious about what happened.",
        "I'm terrified of what might happen next.",
        "I love you so much, you mean everything to me.",
        "Wait what?! I can't believe that just happened!",
    ]
    for text in samples:
        result   = detect_emotion(model, tokenizer, text)
        strategy = get_response_strategy(result["emotion"])
        print(f"\n  Text      : {text}")
        print(f"  Emotion   : {result['emotion']} ({result['confidence']:.0%})")
        print(f"  Strategy  : {strategy}")