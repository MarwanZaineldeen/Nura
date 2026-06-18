from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "module_4_rag_retrieval"
MIN_RESPONSE_WORDS = 25


def clean_text(text: str) -> str:
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    df = dataset.to_pandas().rename(columns={"Context": "context", "Response": "response"})

    df["context"] = df["context"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)
    df["context_words"] = df["context"].str.split().str.len()
    df["response_words"] = df["response"].str.split().str.len()

    quality_checks = {
        "raw_rows": int(len(df)),
        "empty_questions": int((df["context"] == "").sum()),
        "empty_answers": int((df["response"] == "").sum()),
        "very_short_answers": int((df["response_words"] < MIN_RESPONSE_WORDS).sum()),
        "exact_duplicate_rows": int(df.duplicated(subset=["context", "response"]).sum()),
        "unique_questions_before_cleaning": int(df["context"].nunique()),
        "unique_answers_before_cleaning": int(df["response"].nunique()),
    }

    clean_df = df[(df["context"] != "") & (df["response"] != "")].copy()
    clean_df = clean_df[clean_df["response_words"] >= MIN_RESPONSE_WORDS]
    clean_df = clean_df.drop_duplicates(subset=["context", "response"]).reset_index(drop=True)

    question_group_sizes = clean_df["context"].value_counts()
    clean_df["context_group_size"] = clean_df["context"].map(question_group_sizes)
    clean_df["qa_id"] = [f"amod_qa_{index:04d}" for index in range(1, len(clean_df) + 1)]
    clean_df["source"] = "Amod/mental_health_counseling_conversations"

    clean_df = clean_df[
        [
            "qa_id",
            "source",
            "context",
            "response",
            "context_words",
            "response_words",
            "context_group_size",
        ]
    ].rename(
        columns={
            "context": "question",
            "response": "answer",
            "context_words": "question_words",
            "response_words": "answer_words",
            "context_group_size": "question_group_size",
        }
    )

    clean_path = PROCESSED_DIR / "amod_clean_qa.json"
    summary_path = REPORT_DIR / "amod_dataset_summary.json"

    clean_path.write_text(
        json.dumps(clean_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "dataset": "Amod/mental_health_counseling_conversations",
        "raw_rows": int(len(df)),
        "clean_rows": int(len(clean_df)),
        "quality_checks": quality_checks,
        "unique_questions": int(clean_df["question"].nunique()),
        "unique_answers": int(clean_df["answer"].nunique()),
        "repeated_question_count": int((question_group_sizes > 1).sum()),
        "rows_with_repeated_question": int((clean_df["question_group_size"] > 1).sum()),
        "min_question_words": int(clean_df["question_words"].min()),
        "max_question_words": int(clean_df["question_words"].max()),
        "average_question_words": round(float(clean_df["question_words"].mean()), 2),
        "min_answer_words": int(clean_df["answer_words"].min()),
        "max_answer_words": int(clean_df["answer_words"].max()),
        "average_answer_words": round(float(clean_df["answer_words"].mean()), 2),
        "minimum_answer_words_filter": MIN_RESPONSE_WORDS,
        "top_repeated_question_group_sizes": [int(value) for value in question_group_sizes.head(10).to_list()],
        "main_observation": (
            "The dataset is Q&A/counseling-case shaped, with many repeated "
            "questions and multiple possible answers per user concern."
        ),
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved clean dataset to {clean_path}")
    print(f"Saved summary report to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
