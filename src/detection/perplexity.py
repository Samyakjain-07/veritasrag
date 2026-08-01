"""Perplexity-based poison scoring.

PoisonedRAG-style adversarial passages tend to read as stiffer, more
formulaic text than natural corpus passages (see the templates in
`eval/synthetic_poison_generator.py`), which shows up as elevated perplexity
under a small causal language model. Empirically validated against the
Phase 1 manifest: clean chunks scored a mean/median perplexity of 37.6/31.1
under `distilgpt2` vs. 99.9/96.5 for poisoned chunks (ROC-AUC 0.844) — the
strongest single signal of the three detectors in this module.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.detection.config import DEFAULT_CONFIG, DetectionConfig


class PerplexityScorer:
    """Wraps a small HF causal LM to score text by perplexity under it."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_CONFIG.perplexity_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        self._max_length = self._model.config.max_position_embeddings

    @torch.no_grad()
    def compute(self, text: str) -> float:
        """Return the raw perplexity of `text` under the wrapped causal LM."""
        encoded = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self._max_length
        )
        input_ids = encoded["input_ids"]
        outputs = self._model(input_ids, labels=input_ids)
        return float(torch.exp(outputs.loss))

    def compute_batch(self, texts: list[str]) -> list[float]:
        """Return raw perplexity for each text in `texts`, in order."""
        return [self.compute(text) for text in texts]


def normalize_perplexity(
    raw_scores: list[float] | np.ndarray, config: DetectionConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Clip-and-scale raw perplexity values to [0, 1] using fixed config anchors.

    Uses fixed anchors rather than batch min-max so a chunk's normalized score
    doesn't depend on what else happened to be in the same batch — important
    once this is called per-query on small candidate sets instead of a full
    corpus eval batch.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    span = config.perplexity_high_anchor - config.perplexity_low_anchor
    normalized = (scores - config.perplexity_low_anchor) / span
    return np.clip(normalized, 0.0, 1.0)
