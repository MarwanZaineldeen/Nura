from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.retrieval.retrieval_engine import RetrievalEngine


REPORT_DIR = PROJECT_ROOT / "reports" / "module_4_rag_retrieval"
OLD_COLLECTION = "mental_health_rag"
NEW_COLLECTION = "mental_health_rag_v2"

QUERY_SUITE = [
    "What can help during a panic attack at work?",
    "How can I stop worrying at night?",
    "What should I do when I keep seeking reassurance?",
    "How can I improve low self-esteem?",
    "What are practical ways to manage procrastination?",
    "How can I calm health anxiety?",
    "What can help with social anxiety before meeting people?",
    "How do I handle perfectionism when it makes me stuck?",
]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    top_scores = [item["top_score"] for item in results if item["top_score"] is not None]
    top_words = [item["top_word_count"] for item in results if item["top_word_count"] is not None]
    unique_titles = [item["unique_titles_in_top_5"] for item in results]

    return {
        "query_count": len(results),
        "average_top_score": round(statistics.mean(top_scores), 4) if top_scores else None,
        "average_top_word_count": round(statistics.mean(top_words), 1) if top_words else None,
        "average_unique_titles_in_top_5": round(statistics.mean(unique_titles), 2) if unique_titles else None,
    }


def run_collection(engine: RetrievalEngine, collection_name: str, top_k: int) -> list[dict[str, Any]]:
    engine.collection_name = collection_name
    rows = []

    for query in QUERY_SUITE:
        results = engine.search(query, source="cci", top_k=top_k)
        titles = [item.get("title") for item in results if item.get("title")]
        top = results[0] if results else {}
        rows.append(
            {
                "query": query,
                "top_score": top.get("score"),
                "top_title": top.get("title"),
                "top_topic": top.get("topic"),
                "top_word_count": (top.get("metadata") or {}).get("word_count"),
                "unique_titles_in_top_5": len(set(titles)),
                "top_results": [
                    {
                        "rank": item["rank"],
                        "score": item["score"],
                        "title": item.get("title"),
                        "topic": item.get("topic"),
                        "word_count": (item.get("metadata") or {}).get("word_count"),
                    }
                    for item in results
                ],
            }
        )

    return rows


def write_markdown(report: dict[str, Any], path: Path) -> None:
    old_summary = report["summary"][OLD_COLLECTION]
    new_summary = report["summary"][NEW_COLLECTION]
    lines = [
        "# CCI Chunking Strategy Comparison",
        "",
        "This report compares the previous CCI vector index with the current structure-aware CCI index using the same retrieval queries.",
        "",
        "## Collections",
        f"- Previous index: `{OLD_COLLECTION}`",
        f"- Current index: `{NEW_COLLECTION}`",
        "",
        "## Summary",
        f"- Previous average top score: `{old_summary['average_top_score']}`",
        f"- Current average top score: `{new_summary['average_top_score']}`",
        f"- Previous average top chunk size: `{old_summary['average_top_word_count']}` words",
        f"- Current average top chunk size: `{new_summary['average_top_word_count']}` words",
        f"- Previous average title diversity in top 5: `{old_summary['average_unique_titles_in_top_5']}`",
        f"- Current average title diversity in top 5: `{new_summary['average_unique_titles_in_top_5']}`",
        "",
        "## Recommendation",
        "Use `mental_health_rag_v2` as the production index. The current CCI chunks are bounded, easier for the LLM to use, and avoid sending oversized worksheet-sized passages into generation.",
        "",
        "Cosine scores are retrieval similarity signals, not correctness probabilities. The final quality check should combine this report with manual answer review.",
        "",
        "## Query-Level Results",
    ]

    for old_row, new_row in zip(report["collections"][OLD_COLLECTION], report["collections"][NEW_COLLECTION]):
        lines.extend(
            [
                "",
                f"### {old_row['query']}",
                f"- Previous top result: `{old_row['top_title']}` / `{old_row['top_topic']}` / score `{old_row['top_score']}` / `{old_row['top_word_count']}` words",
                f"- Current top result: `{new_row['top_title']}` / `{new_row['top_topic']}` / score `{new_row['top_score']}` / `{new_row['top_word_count']}` words",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new CCI retrieval chunking strategies.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    engine = RetrievalEngine(collection_name=NEW_COLLECTION)

    collections = {
        OLD_COLLECTION: run_collection(engine, OLD_COLLECTION, args.top_k),
        NEW_COLLECTION: run_collection(engine, NEW_COLLECTION, args.top_k),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "source_filter": "cci",
        "collections": collections,
        "summary": {name: summarize_results(rows) for name, rows in collections.items()},
        "recommendation": "Use mental_health_rag_v2 for production because it uses cleaner, bounded, structure-aware CCI chunks.",
    }

    json_path = REPORT_DIR / "chunking_strategy_comparison.json"
    md_path = REPORT_DIR / "chunking_strategy_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
