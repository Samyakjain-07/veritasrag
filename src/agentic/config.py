"""Centralized, sweepable configuration for the Phase 8 agentic-RAG layer
(`src/agentic/planner.py`, `src/agentic/graph.py`).

Mirrors `src/retrieval/config.py`/`src/generation/config.py`'s exact pattern:
every field is a `field(default_factory=...)` (re-read from the environment
per-instantiation, never a bare `os.getenv(...)` default expression), so a
Phase 9 cost-accuracy sweep can construct alternate instances directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_int(var: str, default: str) -> int:
    return int(os.getenv(var, default))


def _env_float(var: str, default: str) -> float:
    return float(os.getenv(var, default))


@dataclass(frozen=True)
class AgenticConfig:
    """All tunable knobs for the Phase 8 planner -> parallel retriever agents
    -> synthesizer graph (`src/agentic/graph.py`)."""

    agentic_max_subqueries: int = field(
        default_factory=lambda: _env_int("AGENTIC_MAX_SUBQUERIES", "3")
    )
    agentic_planner_temperature: float = field(
        default_factory=lambda: _env_float("AGENTIC_PLANNER_TEMPERATURE", "0.0")
    )
    agentic_min_subquery_chars: int = field(
        default_factory=lambda: _env_int("AGENTIC_MIN_SUBQUERY_CHARS", "4")
    )


DEFAULT_CONFIG = AgenticConfig()
