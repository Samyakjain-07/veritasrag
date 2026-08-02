"""Centralized, sweepable configuration for Phase 5 generation + faithfulness.

Mirrors `src/detection/config.py` and `src/retrieval/config.py`'s exact
pattern: every field is a `field(default_factory=...)` (re-read from the
environment per-instantiation, never a bare `os.getenv(...)` default
expression), so a Phase 9 cost-accuracy sweep can construct alternate
instances directly in-process. Both configs live in one file, same as
`DetectionConfig` centralizing every Phase 3 submodule's knobs -- `generate.py`
and `faithfulness.py` are the two Phase 5 submodules here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_str(var: str, default: str) -> str:
    return os.getenv(var, default)


def _env_float(var: str, default: str) -> float:
    return float(os.getenv(var, default))


def _env_int(var: str, default: str) -> int:
    return int(os.getenv(var, default))


DEFAULT_SYSTEM_PROMPT = (
    "You are VeritasRAG, a careful assistant that answers strictly from the "
    "provided, trust-gated context. Cite the chunk id(s) that support each "
    "factual claim in square brackets, e.g. [chunk-00123]. If the context "
    "does not contain enough information to answer, say so explicitly "
    "instead of guessing or using outside knowledge."
)


@dataclass(frozen=True)
class GenerationConfig:
    """All tunable knobs for Phase 5 answer generation (src/generation/generate.py)."""

    generation_max_context_chunks: int = field(
        default_factory=lambda: _env_int("GENERATION_MAX_CONTEXT_CHUNKS", "5")
    )
    generation_max_chars_per_chunk: int = field(
        default_factory=lambda: _env_int("GENERATION_MAX_CHARS_PER_CHUNK", "1500")
    )
    generation_temperature: float = field(
        default_factory=lambda: _env_float("GENERATION_TEMPERATURE", "0.0")
    )
    generation_system_prompt: str = field(
        default_factory=lambda: _env_str("GENERATION_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    )
    generation_no_context_answer: str = field(
        default_factory=lambda: _env_str(
            "GENERATION_NO_CONTEXT_ANSWER",
            "I don't have enough verified information in the trusted context to answer that.",
        )
    )


DEFAULT_GENERATION_CONFIG = GenerationConfig()


@dataclass(frozen=True)
class FaithfulnessConfig:
    """All tunable knobs for the Phase 5 NLI faithfulness check
    (src/generation/faithfulness.py)."""

    faithfulness_nli_model_name: str = field(
        default_factory=lambda: _env_str(
            "FAITHFULNESS_NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-xsmall"
        )
    )
    faithfulness_entailment_threshold: float = field(
        default_factory=lambda: _env_float("FAITHFULNESS_ENTAILMENT_THRESHOLD", "0.5")
    )
    faithfulness_min_claim_chars: int = field(
        default_factory=lambda: _env_int("FAITHFULNESS_MIN_CLAIM_CHARS", "8")
    )


DEFAULT_FAITHFULNESS_CONFIG = FaithfulnessConfig()
