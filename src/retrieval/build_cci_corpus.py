from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "CCI"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cci_information_sheets.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "module_4_rag_retrieval" / "cci_corpus_summary.json"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 80

SENSITIVE_TOPICS = {
    "Bipolar",
    "Body Dysmorphia",
    "Eating Disorders",
}

FOOTER_PATTERNS = [
    r"this document is for information purposes only",
    r"full disclaimer and copyright statement",
    r"cci\.health\.wa\.gov\.au",
    r"centre for clinical interventions",
    r"psychotherapy\s*research\s*training",
    r"pg\s+\d+\s+of\s+\d+",
    r"nov\s+\d{4}\s+v\d",
]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def clean_title(pdf_path: Path) -> str:
    title = pdf_path.stem
    title = re.sub(r"^(.*?)Information Sheet\s*-\s*\d+\s*-\s*", "", title, flags=re.I)
    title = re.sub(r"^Info[-\s]*", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title)
    return title.strip(" -_")


def clean_block_text(text: str) -> str:
    text = text.replace("\u2022", "-")
    text = text.replace("\uf0b7", "-")
    text = text.replace("\u00a0", " ")
    text = text.replace("h\u01a9p", "http")
    text = text.replace("\u01a9", "tt")

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) <= 2:
            continue

        lower = line.lower()
        if any(re.search(pattern, lower) for pattern in FOOTER_PATTERNS):
            continue

        lines.append(line)

    return " ".join(lines).strip()


def normalize_text(text: str) -> str:
    text = re.sub(
        r"This document is for informa\S* purposes only.*?(?:such informa\S*on\.|$)",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"Centre for\s+linical\s+nterventions", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_text(page: fitz.Page) -> str:
    width = page.rect.width
    height = page.rect.height
    midpoint = width / 2

    full_width_blocks: list[tuple[float, float, str]] = []
    left_blocks: list[tuple[float, float, str]] = []
    right_blocks: list[tuple[float, float, str]] = []

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = block
        text = clean_block_text(text)
        if not text or y0 > height * 0.88:
            continue

        block_width = x1 - x0
        block_key = (y0, x0, text)

        if y0 < height * 0.12 or block_width > width * 0.65:
            full_width_blocks.append(block_key)
        elif x0 < midpoint:
            left_blocks.append(block_key)
        else:
            right_blocks.append(block_key)

    ordered_blocks = sorted(full_width_blocks) + sorted(left_blocks) + sorted(right_blocks)
    return " ".join(text for _, _, text in ordered_blocks)


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    return normalize_text(" ".join(extract_page_text(page) for page in doc))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            sentence_end = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if sentence_end > start + int(chunk_size * 0.6):
                end = sentence_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)

    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_SIZE:
        chunks[-2] = normalize_text(f"{chunks[-2]} {chunks[-1]}")
        chunks.pop()

    return chunks


def build_documents(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    topic = pdf_path.parent.name
    document_title = clean_title(pdf_path)
    text = extract_pdf_text(pdf_path)
    base_id = f"cci_{slugify(topic)}_{slugify(document_title)}"
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    documents = []
    for index, chunk in enumerate(chunks, start=1):
        words = re.findall(r"\b\w+\b", chunk)
        documents.append(
            {
                "chunk_id": f"{base_id}_chunk_{index:03d}",
                "document_id": base_id,
                "source": "Centre for Clinical Interventions",
                "topic": topic,
                "sensitivity": "clinical_sensitive" if topic in SENSITIVE_TOPICS else "general_self_help",
                "document_title": document_title,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "text": chunk,
                "word_count": len(words),
            }
        )

    return documents


def build_corpus(input_dir: Path, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    documents = []
    for path in sorted(input_dir.rglob("*.pdf")):
        documents.extend(build_documents(path, chunk_size, chunk_overlap))
    return documents


def save_report(documents: list[dict[str, Any]], report_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    topic_counts = Counter(document["topic"] for document in documents)
    sensitivity_counts = Counter(document["sensitivity"] for document in documents)
    source_document_count = len({document["document_id"] for document in documents})
    word_counts = [document["word_count"] for document in documents]

    report = {
        "source": "Centre for Clinical Interventions",
        "source_document_count": source_document_count,
        "chunk_count": len(documents),
        "chunk_size_characters": chunk_size,
        "chunk_overlap_characters": chunk_overlap,
        "topic_counts": dict(sorted(topic_counts.items())),
        "sensitivity_counts": dict(sorted(sensitivity_counts.items())),
        "total_words": sum(word_counts),
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "average_words": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        "output_format": "fixed-size overlapping text chunks",
        "fields": [
            "chunk_id",
            "document_id",
            "source",
            "topic",
            "sensitivity",
            "document_title",
            "chunk_index",
            "chunk_count",
            "text",
            "word_count",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean JSON corpus from CCI information-sheet PDFs.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, type=Path)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, type=Path)
    parser.add_argument("--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int)
    parser.add_argument("--chunk-overlap", default=DEFAULT_CHUNK_OVERLAP, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = build_corpus(args.input_dir, args.chunk_size, args.chunk_overlap)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    save_report(documents, args.report_path, args.chunk_size, args.chunk_overlap)

    print(f"Saved {len(documents)} CCI documents to {args.output_path}")
    print(f"Saved corpus summary to {args.report_path}")


if __name__ == "__main__":
    main()
