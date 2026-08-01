"""Centralized, sweepable configuration for the Phase 3 poison-detection module.

Every threshold, fusion weight, and model name used across
`perplexity.py`/`cluster_anomaly.py`/`classifier.py`/`pipeline.py`/`llm_judge.py`
lives on `DetectionConfig` instead of being scattered as per-module constants.
Each field still defaults from an environment variable (same ergonomics as the
rest of the codebase's `os.getenv("VAR", default)` pattern), but centralizing
them means a Phase 9 cost-accuracy sweep can construct alternate
`DetectionConfig(...)` instances directly in-process instead of juggling
`os.environ` + module reloads.

`OPENAI_API_KEY` is deliberately never a field here: this object gets
serialized into eval-result JSON artifacts, and a secret has no business in a
report file. `llm_judge.py` reads it straight from the environment instead.
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


@dataclass(frozen=True)
class DetectionConfig:
    """All tunable knobs for the poison-detection pipeline, sweepable as a unit.

    Each field's default is a `field(default_factory=...)` (not a plain
    expression) so it's re-read from the environment on every instantiation,
    not just once at class-definition time -- a plain `os.getenv(...)`
    default expression would otherwise only ever be evaluated at import.
    """

    # Perplexity (src/detection/perplexity.py)
    perplexity_model_name: str = field(
        default_factory=lambda: _env_str("PERPLEXITY_MODEL_NAME", "distilgpt2")
    )
    perplexity_low_anchor: float = field(
        default_factory=lambda: _env_float("PERPLEXITY_LOW_ANCHOR", "20.0")
    )
    perplexity_high_anchor: float = field(
        default_factory=lambda: _env_float("PERPLEXITY_HIGH_ANCHOR", "120.0")
    )
    perplexity_threshold: float = field(
        default_factory=lambda: _env_float("PERPLEXITY_THRESHOLD", "60.0")
    )

    # Cluster anomaly (src/detection/cluster_anomaly.py)
    cluster_knn_k: int = field(default_factory=lambda: _env_int("CLUSTER_KNN_K", "5"))
    cluster_dbscan_eps: float = field(
        default_factory=lambda: _env_float("CLUSTER_DBSCAN_EPS", "0.25")
    )
    cluster_dbscan_min_samples: int = field(
        default_factory=lambda: _env_int("CLUSTER_DBSCAN_MIN_SAMPLES", "2")
    )
    cluster_knn_weight: float = field(
        default_factory=lambda: _env_float("CLUSTER_KNN_WEIGHT", "0.5")
    )
    cluster_dbscan_weight: float = field(
        default_factory=lambda: _env_float("CLUSTER_DBSCAN_WEIGHT", "0.5")
    )
    cluster_anomaly_threshold: float = field(
        default_factory=lambda: _env_float("CLUSTER_ANOMALY_THRESHOLD", "0.5")
    )

    # Classifier (src/detection/classifier.py)
    classifier_seed: int = field(default_factory=lambda: _env_int("CLASSIFIER_SEED", "42"))
    classifier_n_splits: int = field(default_factory=lambda: _env_int("CLASSIFIER_N_SPLITS", "5"))
    classifier_threshold: float = field(
        default_factory=lambda: _env_float("CLASSIFIER_THRESHOLD", "0.5")
    )

    # Fusion (src/detection/pipeline.py). Defaults recalibrated from the
    # Phase 3 eval run: the classifier already stacks perplexity + cluster +
    # embeddings as features and dominated both standalone signals (CV F1
    # 0.926 vs. 0.558 / 0.360), so it gets the majority weight here. Still a
    # coarse manual calibration from one eval run, not a proper sweep --
    # Phase 9's cost-accuracy frontier report is where these get tuned
    # properly.
    fusion_perplexity_weight: float = field(
        default_factory=lambda: _env_float("FUSION_PERPLEXITY_WEIGHT", "0.25")
    )
    fusion_cluster_weight: float = field(
        default_factory=lambda: _env_float("FUSION_CLUSTER_WEIGHT", "0.15")
    )
    fusion_classifier_weight: float = field(
        default_factory=lambda: _env_float("FUSION_CLASSIFIER_WEIGHT", "0.60")
    )
    poison_score_threshold: float = field(
        default_factory=lambda: _env_float("POISON_SCORE_THRESHOLD", "0.5")
    )

    # LLM-as-judge baseline (src/detection/llm_judge.py)
    openai_judge_model: str = field(
        default_factory=lambda: _env_str("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")
    )
    judge_threshold: float = field(default_factory=lambda: _env_float("JUDGE_THRESHOLD", "0.5"))
    judge_max_workers: int = field(default_factory=lambda: _env_int("JUDGE_MAX_WORKERS", "6"))


DEFAULT_CONFIG = DetectionConfig()
