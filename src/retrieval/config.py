"""Centralized, sweepable configuration for the Phase 4 retrieval policy engine.

Mirrors `src/detection/config.py`'s pattern exactly: every threshold lives on
`PolicyConfig` as a `field(default_factory=...)` (re-read from the
environment per-instantiation, never a bare `os.getenv(...)` default
expression -- that only evaluates once at class-definition time, which was a
real bug the first time this pattern was introduced in Phase 3), so a Phase 9
cost-accuracy sweep can construct alternate instances directly in-process.

`PolicyConfig`'s poison-score thresholds are deliberately independent of
`DetectionConfig.poison_score_threshold`: detection's threshold answers "is
this chunk poisoned for eval purposes," while these answer "what should
retrieval do about it," a different, trust-tier-conditional question that
needs to sweep on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.vectorstore.schema import COLLECTION_NAME

load_dotenv()


def _env_str(var: str, default: str) -> str:
    return os.getenv(var, default)


def _env_float(var: str, default: str) -> float:
    return float(os.getenv(var, default))


def _env_int(var: str, default: str) -> int:
    return int(os.getenv(var, default))


def _env_bool(var: str, default: str) -> bool:
    return os.getenv(var, default).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class PolicyConfig:
    """All tunable knobs for the retrieval policy engine, sweepable as a unit.

    Each field's default is a `field(default_factory=...)` (not a plain
    expression) so it's re-read from the environment on every instantiation,
    not just once at class-definition time.
    """

    # Retrieval sizing
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", "5"))
    retrieval_overfetch_factor: float = field(
        default_factory=lambda: _env_float("RETRIEVAL_OVERFETCH_FACTOR", "3.0")
    )
    retrieval_collection_name: str = field(
        default_factory=lambda: _env_str("RETRIEVAL_COLLECTION_NAME", COLLECTION_NAME)
    )

    # Trust-tier categorical policy. QUARANTINED is a categorical drop by
    # default, not a numeric band -- this is the only override.
    allow_quarantined_tier: bool = field(
        default_factory=lambda: _env_bool("POLICY_ALLOW_QUARANTINED_TIER", "false")
    )

    # Per-tier poison_score bands (src/retrieval/policy_engine.py::_gate).
    # Verified is strictly more permissive than unverified: trust_tier and
    # poison_score are meant to compound, not let verified bypass scoring.
    poison_downweight_threshold_verified: float = field(
        default_factory=lambda: _env_float("POLICY_POISON_DOWNWEIGHT_VERIFIED", "0.55")
    )
    poison_hard_drop_threshold_verified: float = field(
        default_factory=lambda: _env_float("POLICY_POISON_HARDDROP_VERIFIED", "0.85")
    )
    poison_downweight_threshold_unverified: float = field(
        default_factory=lambda: _env_float("POLICY_POISON_DOWNWEIGHT_UNVERIFIED", "0.35")
    )
    poison_hard_drop_threshold_unverified: float = field(
        default_factory=lambda: _env_float("POLICY_POISON_HARDDROP_UNVERIFIED", "0.65")
    )
    downweight_floor_multiplier: float = field(
        default_factory=lambda: _env_float("POLICY_DOWNWEIGHT_FLOOR_MULTIPLIER", "0.2")
    )

    # Retrieval-frequency anomaly tracker (src/retrieval/policy_engine.py::FrequencyTracker)
    frequency_zscore_threshold: float = field(
        default_factory=lambda: _env_float("FREQUENCY_ZSCORE_THRESHOLD", "3.0")
    )
    frequency_min_samples: int = field(
        default_factory=lambda: _env_int("FREQUENCY_MIN_SAMPLES", "20")
    )
    frequency_hard_block: bool = field(
        default_factory=lambda: _env_bool("FREQUENCY_HARD_BLOCK", "false")
    )

    # Cluster-anomaly small-batch stabilization. A fixed-size random corpus
    # sample passed as DetectionPipeline.score_chunks's `reference_texts` to
    # partially stabilize the k-NN density sub-signal on small per-query
    # candidate batches -- see policy_engine.py's module docstring for why
    # this only partially, not fully, mitigates the underlying train/inference
    # scale mismatch.
    cluster_reference_sample_size: int = field(
        default_factory=lambda: _env_int("CLUSTER_REFERENCE_SAMPLE_SIZE", "128")
    )

    # Exit-criteria thresholds (PLAN.md Phase 4: "the minor-project gate"),
    # config-driven like every other threshold here so eval/run_eval.py never
    # hardcodes them.
    exit_criteria_asr_reduction_pp_min: float = field(
        default_factory=lambda: _env_float("EXIT_CRITERIA_ASR_REDUCTION_PP_MIN", "30.0")
    )
    exit_criteria_precision_loss_relative_pct_max: float = field(
        default_factory=lambda: _env_float("EXIT_CRITERIA_PRECISION_LOSS_RELATIVE_PCT_MAX", "15.0")
    )


DEFAULT_CONFIG = PolicyConfig()
