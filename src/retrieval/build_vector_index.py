from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

try:
    from src.retrieval.env_utils import load_env_file
except ModuleNotFoundError:
    from env_utils import load_env_file

load_env_file()

try:
    from src.retrieval.embedding_model import E5Embedder, MODEL_NAME
except ModuleNotFoundError:
    from embedding_model import E5Embedder, MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "module_4_rag_retrieval"

CCI_PATH = PROCESSED_DIR / "cci_information_sheets.json"
AMOD_PATH = PROCESSED_DIR / "amod_clean_qa.json"
REPORT_PATH = REPORT_DIR / "retrieval_index_summary.json"

DEFAULT_COLLECTION_NAME = "mental_health_rag_v2"
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "2"))


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_cci_record(item: dict[str, Any]) -> dict[str, Any]:
    text = item["text"]
    title = item["document_title"]
    topic = item["topic"]

    return {
        "id": item["chunk_id"],
        "record_id": item["chunk_id"],
        "source_type": "cci",
        "source": item["source"],
        "title": title,
        "topic": topic,
        "retrieval_text": f"Topic: {topic}. Title: {title}. {text}",
        "display_text": text,
        "metadata": {
            "document_id": item["document_id"],
            "chunk_index": item["chunk_index"],
            "chunk_count": item["chunk_count"],
            "sensitivity": item["sensitivity"],
            "word_count": item["word_count"],
        },
    }


def build_amod_record(item: dict[str, Any]) -> dict[str, Any]:
    question = item["question"]
    answer = item["answer"]

    return {
        "id": item["qa_id"],
        "record_id": item["qa_id"],
        "source_type": "amod",
        "source": item["source"],
        "title": "Counseling Q&A",
        "topic": "counseling_case",
        "retrieval_text": f"Question: {question} Answer: {answer}",
        "display_text": answer,
        "metadata": {
            "question": question,
            "question_words": item["question_words"],
            "answer_words": item["answer_words"],
            "question_group_size": item["question_group_size"],
        },
    }


def build_records() -> list[dict[str, Any]]:
    cci_records = [build_cci_record(item) for item in load_json(CCI_PATH)]
    amod_records = [build_amod_record(item) for item in load_json(AMOD_PATH)]
    return cci_records + amod_records


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise EnvironmentError("Set QDRANT_URL and QDRANT_API_KEY before building the index.")

    return QdrantClient(url=url, api_key=api_key, timeout=60, check_compatibility=False)


def build_points(records: list[dict[str, Any]], embeddings: list[list[float]], start_id: int) -> list[PointStruct]:
    points = []
    for point_id, (record, vector) in enumerate(zip(records, embeddings), start=start_id):
        payload = {key: value for key, value in record.items() if key != "retrieval_text"}
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
    return points


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool) -> None:
    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="source_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def upload_records(
    client: QdrantClient,
    collection_name: str,
    records: list[dict[str, Any]],
    embedder: E5Embedder,
) -> None:
    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start : start + BATCH_SIZE]
        passages = [f"passage: {record['retrieval_text']}" for record in batch_records]
        embeddings = embedder.encode(passages, batch_size=BATCH_SIZE)
        points = build_points(batch_records, embeddings.tolist(), start_id=start + 1)
        client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded {min(start + BATCH_SIZE, len(records))}/{len(records)} records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Qdrant Cloud retrieval index.")
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME))
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection before upload.")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = get_qdrant_client()
    except EnvironmentError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    records = build_records()

    embedder = E5Embedder(MODEL_NAME)
    ensure_collection(client, args.collection, vector_size=embedder.dimension, recreate=args.recreate)
    upload_records(client, args.collection, records, embedder)

    source_counts = Counter(record["source_type"] for record in records)
    report = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": embedder.dimension,
        "vector_database": "Qdrant Cloud",
        "similarity_metric": "cosine_similarity",
        "collection_name": args.collection,
        "record_count": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "batch_size": BATCH_SIZE,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "payload_indexes": ["source_type"],
        "recreated_collection": args.recreate,
        "output_note": "Embeddings are normalized and stored in Qdrant with cosine distance.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Uploaded {len(records)} points to Qdrant collection: {args.collection}")
    print(f"Saved index report to {REPORT_PATH}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
