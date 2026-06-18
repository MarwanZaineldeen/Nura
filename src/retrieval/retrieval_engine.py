from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue

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
DEFAULT_COLLECTION_NAME = "mental_health_rag"
SOURCE_OPTIONS = {"both", "cci", "amod"}


class RetrievalEngine:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        collection_name: str | None = None,
    ) -> None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise EnvironmentError("Set QDRANT_URL and QDRANT_API_KEY before searching.")

        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60, check_compatibility=False)
        self.embedder = E5Embedder(model_name)

    def search(self, query: str, source: str = "both", top_k: int = 5) -> list[dict[str, Any]]:
        source = source.lower()
        if source not in SOURCE_OPTIONS:
            raise ValueError("source must be one of: both, cci, amod.")

        query_vector = self.embedder.encode([f"query: {query}"])[0].tolist()
        points = self._query_points(query_vector, source, top_k)

        return [self._format_result(point, rank + 1) for rank, point in enumerate(points)]

    def _query_points(self, query_vector: list[float], source: str, limit: int) -> list[Any]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._source_filter(source),
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def _source_filter(self, source: str) -> Filter | None:
        if source == "both":
            return None
        return Filter(must=[FieldCondition(key="source_type", match=MatchValue(value=source))])

    def _format_result(self, point: Any, rank: int) -> dict[str, Any]:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        display_text = payload.get("display_text")

        if payload.get("source_type") == "amod" and metadata.get("question"):
            display_text = f"Question: {metadata['question']}\n\nAnswer: {display_text}"

        return {
            "rank": rank,
            "score": round(float(point.score), 4),
            "id": payload.get("record_id"),
            "source_type": payload.get("source_type"),
            "source": payload.get("source"),
            "title": payload.get("title"),
            "topic": payload.get("topic"),
            "text": display_text,
            "metadata": metadata,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Qdrant mental-health retrieval index.")
    parser.add_argument("query", help="User question to search for.")
    parser.add_argument("--source", choices=sorted(SOURCE_OPTIONS), default="both")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME))
    args = parser.parse_args()

    try:
        engine = RetrievalEngine(collection_name=args.collection)
    except EnvironmentError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    try:
        results = engine.search(args.query, source=args.source, top_k=args.top_k)
    except (ResponseHandlingException, UnexpectedResponse) as error:
        print(f"Qdrant search error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
