"""Phase 5 faithfulness check: per-claim NLI entailment against the retrieved
context.

Splits a generated answer into sentence-level claims and, for each claim,
scores entailment against every context chunk with a small HF NLI
cross-encoder (`cross-encoder/nli-deberta-v3-xsmall` by default -- picked for
the same CPU-only-box, small-and-fast reasoning as `distilgpt2` in
`src/detection/perplexity.py`). A claim is "grounded" if its best-supporting
chunk's entailment probability clears `config.faithfulness_entailment_threshold`.

The entailment label's index is read from the model's own
`config.id2label` rather than assumed to be a fixed position -- label order
isn't guaranteed across NLI checkpoints, and this repo has already hit a real
bug from exactly this kind of unverified assumption (Phase 3's classifier
pickling under `__main__`, see PROGRESS.md).

This module is reused two ways: `generate.py` calls it post-generation to
flag ungrounded/hallucinated claims for the API response, and
`eval/run_eval.py`'s generation-stage eval reuses the same per-claim
`best_chunk_id` to detect whether a poisoned chunk (via manifest ground
truth) ended up grounding part of the final answer -- i.e. whether a
poisoning attack broke through all the way to generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

from src.generation.config import DEFAULT_FAITHFULNESS_CONFIG, FaithfulnessConfig

# Simple sentence splitter: break after ./!/? followed by whitespace. No nltk
# dependency -- good enough for the templated, citation-bracketed answers
# this pipeline produces; not a general-purpose sentence tokenizer.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# A model sometimes puts its citation bracket(s) after the sentence-ending
# period instead of before it (e.g. "...Naoroji. [chunk-1]"), which the
# splitter above then treats as its own sentence. That fragment asserts
# nothing -- NLI correctly finds no entailment for a bare citation -- so
# without this filter it always scores as a spuriously "ungrounded" claim
# and deflates faithfulness_score. Caught live during Phase 5 API
# verification (see PROGRESS.md), not a hypothetical case.
_CITATION_ONLY_RE = re.compile(r"^(\[[^\[\]]*\]\s*)+$")


@dataclass(frozen=True)
class ContextChunk:
    """A single piece of generation context: a gated chunk's id and text."""

    chunk_id: str
    text: str


@dataclass(frozen=True)
class ClaimFaithfulness:
    """One generated claim's best-supporting context chunk and entailment score."""

    claim: str
    best_chunk_id: str | None
    entailment_score: float
    grounded: bool


@dataclass(frozen=True)
class FaithfulnessReport:
    """Per-claim faithfulness results plus an aggregate score for the answer."""

    claims: list[ClaimFaithfulness]
    faithfulness_score: float  # fraction of claims grounded; 1.0 if there were no claims
    ungrounded_claims: list[str]


def split_claims(
    answer: str, config: FaithfulnessConfig = DEFAULT_FAITHFULNESS_CONFIG
) -> list[str]:
    """Split a generated answer into sentence-level claims.

    Drops fragments shorter than `config.faithfulness_min_claim_chars` (stray
    whitespace) and fragments that are nothing but citation bracket(s) (e.g.
    "[chunk-1]") -- those assert no fact to check faithfulness against.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]
    return [
        s
        for s in sentences
        if len(s) >= config.faithfulness_min_claim_chars and not _CITATION_ONLY_RE.match(s)
    ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class FaithfulnessChecker:
    """Wraps a CrossEncoder NLI model, loaded once and reused across calls."""

    def __init__(self, config: FaithfulnessConfig = DEFAULT_FAITHFULNESS_CONFIG) -> None:
        self.config = config
        self._model = CrossEncoder(config.faithfulness_nli_model_name)
        label2id = {label.lower(): idx for idx, label in self._model.config.id2label.items()}
        self._entailment_index = label2id["entailment"]

    def check(self, answer: str, context: list[ContextChunk]) -> FaithfulnessReport:
        """Score every claim in `answer` against every chunk in `context`.

        Vacuously faithful (score 1.0, no claims) if the answer has no
        claims to check (e.g. an "I don't know" abstention) -- there's
        nothing ungrounded being asserted. Wholly unfaithful (score 0.0) if
        there are claims but no context to ground them in.
        """
        claims = split_claims(answer, self.config)
        if not claims:
            return FaithfulnessReport(claims=[], faithfulness_score=1.0, ungrounded_claims=[])
        if not context:
            return FaithfulnessReport(
                claims=[
                    ClaimFaithfulness(
                        claim=claim, best_chunk_id=None, entailment_score=0.0, grounded=False
                    )
                    for claim in claims
                ],
                faithfulness_score=0.0,
                ungrounded_claims=list(claims),
            )

        pairs = [(chunk.text, claim) for claim in claims for chunk in context]
        logits = self._model.predict(pairs)
        probs = _softmax(np.asarray(logits))
        entailment_scores = probs[:, self._entailment_index].reshape(len(claims), len(context))

        claim_results: list[ClaimFaithfulness] = []
        ungrounded: list[str] = []
        grounded_count = 0
        for i, claim in enumerate(claims):
            best_j = int(np.argmax(entailment_scores[i]))
            best_score = float(entailment_scores[i, best_j])
            grounded = best_score >= self.config.faithfulness_entailment_threshold
            if grounded:
                grounded_count += 1
            else:
                ungrounded.append(claim)
            claim_results.append(
                ClaimFaithfulness(
                    claim=claim,
                    best_chunk_id=context[best_j].chunk_id,
                    entailment_score=best_score,
                    grounded=grounded,
                )
            )

        return FaithfulnessReport(
            claims=claim_results,
            faithfulness_score=grounded_count / len(claims),
            ungrounded_claims=ungrounded,
        )
