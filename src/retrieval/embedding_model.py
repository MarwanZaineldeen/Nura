from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
MAX_LENGTH = 512
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "1"))


class E5Embedder:
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.set_num_threads(TORCH_NUM_THREADS)

        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True).to(self.device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)

    def encode(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**batch)
                pooled = average_pool(output.last_hidden_state, batch["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                embeddings.append(normalized.cpu().numpy().astype("float32"))

        return np.vstack(embeddings)


def average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
