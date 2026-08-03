"""Centralized, sweepable configuration for the Phase 6 unlearning module.

Mirrors `src/detection/config.py`/`src/retrieval/config.py`'s exact pattern:
every field is a `field(default_factory=...)` (re-read from the environment
per-instantiation, never a bare `os.getenv(...)` default expression -- that
only evaluates once at class-definition time, a real bug hit and fixed the
first time this pattern was introduced in Phase 3), so a Phase 9
cost-accuracy sweep can construct alternate instances directly in-process.
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
class UnlearningConfig:
    """All tunable knobs for delete-by-ID, index compaction, receipts, and
    the ghost-vector reconstruction defense, sweepable as a unit.
    """

    # Target collection for src/unlearning/delete.py's CLI. eval/run_eval.py's
    # --stage unlearning always overrides this with its own disposable
    # scratch collection rather than reading this field, so the live corpus
    # is never touched by the eval harness.
    unlearning_collection_name: str = field(
        default_factory=lambda: _env_str("UNLEARNING_COLLECTION_NAME", COLLECTION_NAME)
    )
    delete_wait: bool = field(default_factory=lambda: _env_bool("UNLEARNING_DELETE_WAIT", "true"))

    # Index compaction (src/unlearning/delete.py::force_compaction). Setting
    # both thresholds to their minimum forces Qdrant's optimizer to
    # immediately vacuum any segment holding a just-deleted point rather than
    # waiting for the server's default deleted_threshold=0.2 /
    # vacuum_min_vector_number=1000 to be crossed naturally.
    compaction_deleted_threshold: float = field(
        default_factory=lambda: _env_float("UNLEARNING_COMPACTION_DELETED_THRESHOLD", "0.0")
    )
    # Qdrant's server-side validation rejects anything below 100 here
    # ("value 0 invalid, must be 100 or larger") -- confirmed against the
    # live server, not just client-side docs. 100 is the most aggressive
    # legal setting, still far below the server default of 1000.
    compaction_vacuum_min_vector_number: int = field(
        default_factory=lambda: _env_int("UNLEARNING_COMPACTION_VACUUM_MIN_VECTOR_NUMBER", "100")
    )
    compaction_poll_for_green: bool = field(
        default_factory=lambda: _env_bool("UNLEARNING_COMPACTION_POLL_FOR_GREEN", "true")
    )
    compaction_poll_timeout_s: float = field(
        default_factory=lambda: _env_float("UNLEARNING_COMPACTION_POLL_TIMEOUT_S", "10.0")
    )
    compaction_poll_interval_s: float = field(
        default_factory=lambda: _env_float("UNLEARNING_COMPACTION_POLL_INTERVAL_S", "0.1")
    )
    # Restoring the collection's original optimizer thresholds after forcing
    # compaction matters: without this, one delete call would permanently
    # change the collection's optimizer behavior for every future write.
    compaction_restore_original_config: bool = field(
        default_factory=lambda: _env_bool("UNLEARNING_COMPACTION_RESTORE_ORIGINAL_CONFIG", "true")
    )

    # Ghost-vector defense, check 1: membership-inference self-check
    # (src/unlearning/ghost_vector_defense.py::verify_unrecoverable).
    ghost_defense_search_k: int = field(
        default_factory=lambda: _env_int("GHOST_DEFENSE_SEARCH_K", "10")
    )

    # Ghost-vector defense, check 2: corpus-level reconstruction attack vs. a
    # chance baseline (src/unlearning/ghost_vector_defense.py::attempt_reconstruction).
    # The chance baseline compares the attack's best-neighbor similarity
    # against a percentile (not the mean) of decoy queries' own best-neighbor
    # similarity -- a max-vs-mean comparison would make "attack succeeds"
    # read as true almost always, regardless of real leakage. See that
    # module's docstring for the full reasoning.
    reconstruction_chance_sample_size: int = field(
        default_factory=lambda: _env_int("RECONSTRUCTION_CHANCE_SAMPLE_SIZE", "20")
    )
    reconstruction_chance_percentile: float = field(
        default_factory=lambda: _env_float("RECONSTRUCTION_CHANCE_PERCENTILE", "95.0")
    )
    reconstruction_top_k_diagnostic: int = field(
        default_factory=lambda: _env_int("RECONSTRUCTION_TOP_K_DIAGNOSTIC", "3")
    )

    # Deletion receipts (src/unlearning/receipt.py). New top-level directory,
    # not named in PLAN.md Part 3's repo layout -- mirrors the existing
    # keys/ and models/ top-level generated-artifact convention. Flagged in
    # PROGRESS.md per CLAUDE.md's "say so explicitly" rule.
    receipt_dir: str = field(default_factory=lambda: _env_str("UNLEARNING_RECEIPT_DIR", "receipts"))

    # eval/run_eval.py --stage unlearning
    eval_sample_size: int = field(
        default_factory=lambda: _env_int("UNLEARNING_EVAL_SAMPLE_SIZE", "20")
    )


DEFAULT_CONFIG = UnlearningConfig()
