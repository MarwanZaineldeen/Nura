import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
QDRANT_URL     = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME    = "mental_health_rag"
DENSE_MODEL_NAME   = "all-MiniLM-L6-v2"
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL         = "llama-3.3-70b-versatile"
DENSE_DIM          = 384
SEMANTIC_THRESHOLD = 0.45
MAX_CHUNK_SENTENCES= 8
MIN_CHUNK_SENTENCES= 2
TOP_K_RETRIEVE     = 20
TOP_K_RERANK       = 5
HYBRID_ALPHA       = 0.6
QDRANT_BATCH_SIZE  = 64

print("Loading embedding & reranker models...")
dense_model    = SentenceTransformer(DENSE_MODEL_NAME)
reranker_model = CrossEncoder(RERANKER_MODEL)
groq_client    = Groq(api_key=GROQ_API_KEY)
print("Models loaded.")


def semantic_chunk(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if len(sentences) <= MIN_CHUNK_SENTENCES:
        return [text] if text.strip() else []
    embeddings = dense_model.encode(sentences, show_progress_bar=False)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i-1], embeddings[i]) /
                    (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8))
        if sim >= SEMANTIC_THRESHOLD and len(current) < MAX_CHUNK_SENTENCES:
            current.append(sentences[i])
        else:
            if len(current) >= MIN_CHUNK_SENTENCES:
                chunks.append(" ".join(current))
            elif chunks:
                chunks[-1] += " " + " ".join(current)
            else:
                chunks.append(" ".join(current))
            current = [sentences[i]]
    if current:
        chunk_text = " ".join(current)
        if len(current) >= MIN_CHUNK_SENTENCES:
            chunks.append(chunk_text)
        elif chunks:
            chunks[-1] += " " + chunk_text
        else:
            chunks.append(chunk_text)
    return [c for c in chunks if c.strip()]


def load_and_chunk_dataset(max_rows: int = 2000) -> List[Dict]:
    print("Loading dataset...")
    ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    ds = ds.select(range(min(max_rows, len(ds))))
    print(f"Loaded {len(ds)} conversations.")
    all_chunks = []
    for idx, row in enumerate(ds):
        context  = row.get("Context", "")
        response = row.get("Response", "")
        combined = f"Patient: {context}\n\nCounselor: {response}"
        chunks   = semantic_chunk(combined)
        for c_idx, chunk in enumerate(chunks):
            all_chunks.append({
                "id"             : f"{idx}_{c_idx}",
                "text"           : chunk,
                "source"         : f"conversation_{idx}",
                "context_preview": context[:120],
            })
    print(f"Created {len(all_chunks)} semantic chunks from {len(ds)} conversations.")
    return all_chunks


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def index_chunks(chunks: List[Dict], batch_size: int = QDRANT_BATCH_SIZE):
    qdrant   = get_qdrant_client()
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return
    print(f"Embedding {len(chunks)} chunks...")
    points = []
    for i in range(0, len(chunks), batch_size):
        batch       = chunks[i: i + batch_size]
        batch_texts = [b["text"] for b in batch]
        embeddings  = dense_model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
        for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
            points.append(PointStruct(
                id     = i + j,
                vector = emb.tolist(),
                payload= {
                    "text"           : chunk["text"],
                    "source"         : chunk["source"],
                    "context_preview": chunk["context_preview"],
                    "chunk_id"       : chunk["id"],
                }
            ))
        if (i // batch_size) % 5 == 0:
            print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant.")
    return chunks


def build_bm25_index(chunks: List[Dict]) -> Tuple[BM25Okapi, List[Dict]]:
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)
    print("BM25 keyword index built.")
    return bm25, chunks


QUERY_PARSE_SYSTEM = """You are a query parsing assistant for a mental health Q&A system.
Given a user question, extract:
1. core_query  : the cleaned, concise version of the question (remove filler words)
2. keywords    : list of important domain-specific keywords (emotions, conditions, techniques)
3. is_personal : true if the user is sharing personal distress, false if asking general info

Return ONLY valid JSON, no extra text:
{"core_query": "...", "keywords": ["...", "..."], "is_personal": true/false}"""

def parse_query(user_question: str) -> Dict:
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=200,
            temperature=0.0,
            messages=[
                {"role": "system", "content": QUERY_PARSE_SYSTEM},
                {"role": "user",   "content": user_question},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {
            "core_query" : user_question,
            "keywords"   : user_question.lower().split()[:5],
            "is_personal": True,
        }


def hybrid_retrieve(
    query     : str,
    bm25      : BM25Okapi,
    all_chunks: List[Dict],
    top_k     : int = TOP_K_RETRIEVE,
) -> List[Dict]:
    qdrant        = get_qdrant_client()
    query_emb     = dense_model.encode(query, normalize_embeddings=True).tolist()
    dense_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query          =query_emb,
        limit          =top_k,
        with_payload   =True,
    )
    dense_results = dense_response.points
    dense_scores  = {r.payload["chunk_id"]: r.score for r in dense_results}
    tokenized_query = query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
    bm25_map = {
        all_chunks[i]["id"]: float(bm25_scores_raw[i]) / max_bm25
        for i in np.argsort(bm25_scores_raw)[-top_k:]
        if bm25_scores_raw[i] > 0
    }
    all_ids = set(dense_scores.keys()) | set(bm25_map.keys())
    hybrid  = {}
    for cid in all_ids:
        d = dense_scores.get(cid, 0.0)
        b = bm25_map.get(cid, 0.0)
        hybrid[cid] = HYBRID_ALPHA * d + (1 - HYBRID_ALPHA) * b
    top_ids     = sorted(hybrid, key=hybrid.get, reverse=True)[:top_k]
    id_to_chunk = {c["id"]: c for c in all_chunks}
    return [
        {**id_to_chunk[cid], "hybrid_score": hybrid[cid]}
        for cid in top_ids if cid in id_to_chunk
    ]


def rerank(query: str, candidates: List[Dict], top_k: int = TOP_K_RERANK) -> List[Dict]:
    if not candidates:
        return []
    pairs  = [(query, c["text"]) for c in candidates]
    scores = reranker_model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    for score, chunk in ranked:
        chunk["rerank_score"] = float(score)
    return [chunk for _, chunk in ranked[:top_k]]


RAG_SYSTEM_PROMPT = """You are a compassionate, evidence-based mental health support assistant.
Your role is to provide helpful, empathetic, and accurate responses based on counseling knowledge.

Guidelines:
- Always acknowledge the user's feelings before providing information or advice.
- Base your answer on the provided context from mental health counseling conversations.
- If the context does not contain enough information, say so honestly — do not fabricate.
- Use warm, non-judgmental, supportive language.
- If the user seems in distress, gently suggest professional help where appropriate.
- Keep responses clear, structured, and between 150–350 words.
- Never diagnose. Provide psychoeducation and coping strategies instead."""


def build_rag_prompt(
    user_question    : str,
    parsed_query     : Dict,
    context_chunks   : List[Dict],
    emotion_result   : Optional[Dict] = None,
    emotion_strategy : Optional[str] = None,
    response_language: str = "English",
) -> List[Dict]:
    context_text = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    )
    emotion_note = ""
    if emotion_result and emotion_strategy:
        emotion_note = (
            f"\n\nEmotion detected: {emotion_result['emotion']} "
            f"(confidence {emotion_result['confidence']:.0%}). "
            f"Response strategy: {emotion_strategy}"
        )
    user_content = f"""Question: {user_question}

Parsed core query: {parsed_query.get('core_query', user_question)}
Key topics: {', '.join(parsed_query.get('keywords', []))}
{emotion_note}

Relevant counseling context:
{context_text}

Please provide a supportive, evidence-informed response.
IMPORTANT: You MUST respond in {response_language} language only."""

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def rag_answer(
    user_question    : str,
    bm25             : BM25Okapi,
    all_chunks       : List[Dict],
    emotion_result   : Optional[Dict] = None,
    emotion_strategy : Optional[str] = None,
    response_language: str = "English",
    verbose          : bool = False,
) -> Dict:
    parsed       = parse_query(user_question)
    search_query = parsed.get("core_query", user_question)
    if verbose:
        print(f"  [Query Parser] core='{search_query}' | keywords={parsed.get('keywords')}")
    candidates = hybrid_retrieve(search_query, bm25, all_chunks, top_k=TOP_K_RETRIEVE)
    if verbose:
        print(f"  [Hybrid Retrieval] {len(candidates)} candidates retrieved.")
    top_chunks = rerank(search_query, candidates, top_k=TOP_K_RERANK)
    if verbose:
        print(f"  [Reranker] Top {len(top_chunks)} chunks selected.")
        for i, c in enumerate(top_chunks):
            print(f"    [{i+1}] rerank={c['rerank_score']:.3f} | {c['text'][:80]}...")
    messages = build_rag_prompt(
        user_question, parsed, top_chunks, emotion_result, emotion_strategy, response_language
    )
    response = groq_client.chat.completions.create(
        model      = GROQ_MODEL,
        max_tokens = 512,
        temperature= 0.4,
        messages   = messages,
    )
    answer = response.choices[0].message.content.strip()
    return {
        "question"    : user_question,
        "answer"      : answer,
        "parsed_query": parsed,
        "sources"     : [{"text": c["text"], "source": c["source"]} for c in top_chunks],
    }


def setup_rag_pipeline(max_rows: int = 2000):
    chunks = load_and_chunk_dataset(max_rows=max_rows)
    index_chunks(chunks)
    bm25, all_chunks = build_bm25_index(chunks)
    return bm25, all_chunks


if __name__ == "__main__":
    bm25, all_chunks = setup_rag_pipeline(max_rows=2000)
    test_qs = [
        "I feel so anxious all the time, I can't sleep. What can I do?",
        "How do I stop negative thoughts from taking over?",
        "My friend is going through depression. How can I support them?",
    ]
    print("\n" + "="*60)
    print("RAG PIPELINE DEMO")
    print("="*60)
    for q in test_qs:
        print(f"\nQ: {q}")
        result = rag_answer(q, bm25, all_chunks, verbose=True)
        print(f"\nA: {result['answer']}")
        print("-"*60)