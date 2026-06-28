# Hugging Face Spaces Deployment

This project is prepared for deployment as a Hugging Face Docker Space.

## 1. Login Locally

Run:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Use a Hugging Face token with write access.

## 2. Upload Model Artifacts

Create two Hugging Face model repositories:

```text
your_username/mental-health-language-detector
your_username/mental-health-emotion-detector
```

Upload:

- `src/models/saved_lang_model.pkl` to the language model repository.
- All files inside `src/models/saved_emotion_model/` to the emotion model repository.

The emotion model folder should contain:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

## 3. Create The Space

Create a Hugging Face Space:

```text
SDK: Docker
Visibility: Public or Private
App port: 7860
```

## 4. Add Space Secrets

In the Space settings, add:

```text
GROQ_API_KEY
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
LANGUAGE_MODEL_REPO_ID
LANGUAGE_MODEL_FILENAME
EMOTION_MODEL_ID
```

Recommended values:

```text
LANGUAGE_MODEL_FILENAME=saved_lang_model.pkl
QDRANT_COLLECTION=mental_health_rag_v2
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
EMBEDDING_BATCH_SIZE=2
TORCH_NUM_THREADS=1
```

## 5. Push To The Space Repo

After the Space is created, add it as a Git remote:

```powershell
git remote add space https://huggingface.co/spaces/your_username/your_space_name
git push space main
```

The Dockerfile starts the production app with:

```text
uvicorn src.api_app:app --host 0.0.0.0 --port 7860
```

Open the Space URL after the build finishes.
