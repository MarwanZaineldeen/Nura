# Module 4 - RAG Retrieval

This module builds a multilingual retrieval layer for the mental-health chatbot.

## Retrieval Sources

- `cci`: structure-aware CCI information-sheet chunks with a 400-word maximum.
- `amod`: cleaned counseling Q&A pairs.
- `both`: searches both sources in the same Qdrant collection.

## Embedding Model

The default embedding model is:

```text
intfloat/multilingual-e5-base
```

It is used because the knowledge base is mostly English, while user messages may arrive in other languages. A multilingual embedding model allows cross-lingual retrieval before the chatbot generates a final answer in the user's language.

E5 expects prefixes:

- Queries use `query:`
- Stored passages use `passage:`

## Vector Database

The project uses Qdrant Cloud as the vector database.

Required environment variables:

```powershell
$env:QDRANT_URL="https://your-cluster-url.qdrant.tech"
$env:QDRANT_API_KEY="your_qdrant_api_key"
$env:QDRANT_COLLECTION="mental_health_rag_v2"
```

Optional, if the model is already cached somewhere else:

```powershell
$env:HUGGINGFACE_HUB_CACHE="path_to_your_huggingface_hub_cache"
```

## Build Index

```powershell
.\.venv\Scripts\python.exe src\retrieval\build_vector_index.py --recreate
```

## Compare Chunking

```powershell
.\.venv\Scripts\python.exe src\evaluation\compare_retrieval_chunking.py
```

This writes `chunking_strategy_comparison.json` and `chunking_strategy_comparison.md`, comparing the previous CCI index with the current structure-aware CCI chunks.

## Test Retrieval

```powershell
.\.venv\Scripts\python.exe src\retrieval\retrieval_engine.py "I feel anxious all the time" --source both --top-k 5
```

The retrieval output includes rank, cosine similarity, source, title, topic, text, and metadata. Cosine similarity is a ranking metric, not a probability or percentage confidence.

## FastAPI Deployment

The integrated local deployment is served with FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api_app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Pages:

- `/` production chatbot UI
- `/developer` developer/testing UI with pipeline state

API endpoints:

- `GET /health`
- `POST /chat`

Example request body:

```json
{
  "message": "I feel anxious every night and cannot sleep",
  "source": "both",
  "top_k": 8
}
```
