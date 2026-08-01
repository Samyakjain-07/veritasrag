"""Sentence-embedding wrapper around a HuggingFace `sentence-transformers` model."""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")


class Embedder:
    """Wraps a sentence-transformers model for chunk/query embedding."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality for the loaded model."""
        return self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning an (n, dimension) float32 array."""
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text, returning a (dimension,) float32 array."""
        return self.embed([text])[0]
