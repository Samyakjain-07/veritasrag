"""Centralized, sweepable configuration for Phase 7 observability (tracing).

Mirrors `src/detection/config.py`/`src/retrieval/config.py`/`src/unlearning/
config.py`'s exact pattern: every field is a `field(default_factory=...)`
(re-read from the environment per-instantiation, never a bare
`os.getenv(...)` default expression), so tests/tooling can construct
alternate instances directly in-process (e.g. `tests/conftest.py` pointing
`trace_file_path` at a scratch directory for the whole test session).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_str(var: str, default: str) -> str:
    return os.getenv(var, default)


def _env_bool(var: str, default: str) -> bool:
    return os.getenv(var, default).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class ObservabilityConfig:
    """All tunable knobs for OpenTelemetry span export (src/observability/otel_setup.py)."""

    service_name: str = field(default_factory=lambda: _env_str("OTEL_SERVICE_NAME", "veritasrag"))

    # Where spans are persisted. No external trace backend (Jaeger/Tempo/an
    # OTel collector) exists anywhere in this project's infra -- this file is
    # the only "backend," and the dashboard reads it directly.
    trace_file_path: str = field(
        default_factory=lambda: _env_str("OTEL_TRACE_FILE_PATH", "observability/traces.jsonl")
    )

    # "simple" (default) exports each span synchronously on span-end, so a
    # query's spans are visible in the trace file moments after the request
    # finishes -- the right tradeoff for a local dashboard that wants to feel
    # "live." "batch" exists for completeness/future production use.
    # Validated at `setup_tracing()` call time (raises ValueError on anything
    # else), not here -- this is a plain string field like the rest of this
    # repo's config dataclasses, no existing precedent for enum validation at
    # construction time.
    span_processor: str = field(default_factory=lambda: _env_str("OTEL_SPAN_PROCESSOR", "simple"))

    # Also print spans to stdout, for terminal debugging. Off by default --
    # would otherwise double every span's console noise on top of the file.
    console_export: bool = field(default_factory=lambda: _env_bool("OTEL_CONSOLE_EXPORT", "false"))


DEFAULT_CONFIG = ObservabilityConfig()
