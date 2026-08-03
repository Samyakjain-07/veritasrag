"""OpenTelemetry tracing setup for VeritasRAG.

No external trace backend exists in this project's infra (only Qdrant runs
in `docker-compose.yml` -- no Jaeger/Tempo/OTel-collector). Instead,
`JSONLFileSpanExporter` appends each finished span as one JSON line to a
local file (`ObservabilityConfig.trace_file_path`), which `dashboard/data.py`
reads directly -- that file *is* the trace backend for this project.

`get_tracer(name)` is the only entry point instrumented modules should call;
it lazily calls `setup_tracing()` on first use, mirroring the lazy-singleton
pattern already used for `_default_embedder()`/`_default_qdrant_client()` in
`src/retrieval/policy_engine.py`. Every instrumented call site must call it
as the *first statement inside the function body it instruments* -- never at
module level. pytest imports every test module (and transitively every
`src.*` module it references) during collection, before any fixture (even
session-scoped) runs; a module-level call would permanently lock in the real
`observability/traces.jsonl` path before `tests/conftest.py`'s session-wide
fixture gets a chance to redirect it to a scratch file.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer, format_span_id, format_trace_id

from src.observability.config import DEFAULT_CONFIG, ObservabilityConfig


def _span_to_record(span: ReadableSpan) -> dict:
    """Flatten a finished span into a flat, JSON-safe dict.

    `span.attributes`/`span.resource.attributes` are `BoundedAttributes`
    (a `Mapping`, not a `dict`) -- `json.dumps` can't serialize them
    directly, they must be wrapped in `dict(...)` first. `span.parent` is a
    `SpanContext` (only trace_id/span_id/is_remote), not the parent
    `ReadableSpan` itself -- exactly enough to reconstruct the trace tree by
    joining `parent_span_id` against another row's `span_id` within the same
    `trace_id`.
    """
    ctx = span.context
    parent_span_id = format_span_id(span.parent.span_id) if span.parent is not None else None
    resource_attrs = dict(span.resource.attributes) if span.resource else {}
    return {
        "trace_id": format_trace_id(ctx.trace_id),
        "span_id": format_span_id(ctx.span_id),
        "parent_span_id": parent_span_id,
        "name": span.name,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "status_code": span.status.status_code.name,
        "service_name": resource_attrs.get("service.name"),
        "attributes": dict(span.attributes) if span.attributes else {},
    }


class JSONLFileSpanExporter(SpanExporter):
    """Appends one JSON line per finished span to a local file.

    Thread-safety: `SimpleSpanProcessor.on_end()` calls `export()`
    synchronously on whatever thread ended the span -- FastAPI's sync `def`
    routes run in a threadpool, so concurrent `/query` requests genuinely
    call `export()` from different threads at once. A bare buffered-text-mode
    `write()` isn't safely serialized across threads for this without a lock
    (mirrors `FrequencyTracker._lock` in `src/retrieval/policy_engine.py`).

    Known, documented limitation (same shape as `FrequencyTracker`'s own
    caveat): a `threading.Lock` only protects against races *within one
    process* -- running `uvicorn --workers N>1` would give each worker its
    own independent lock and could interleave writes across processes.
    Out of scope for this phase (single-process local dev/demo).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = [json.dumps(_span_to_record(span)) for span in spans]
        try:
            with self._lock, open(self._path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
                f.flush()
            return SpanExportResult.SUCCESS
        except OSError:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True


_provider: TracerProvider | None = None
_configured = False
_lock = threading.Lock()


def setup_tracing(config: ObservabilityConfig = DEFAULT_CONFIG) -> TracerProvider:
    """Idempotently configure the process-wide `TracerProvider`.

    Only the *first* call's `config` takes effect -- every later call
    (regardless of what `config` it's passed) just returns the already-built
    provider. This matches OpenTelemetry's own global-API contract
    (`trace.set_tracer_provider()` only succeeds once process-wide; later
    calls are silently ignored with a logged warning), and makes the
    provider's lifecycle predictable: whichever code path calls this first
    (a test fixture, the API startup, a CLI entrypoint) wins.
    """
    global _provider, _configured
    with _lock:
        if _configured and _provider is not None:
            return _provider

        if config.span_processor not in ("simple", "batch"):
            raise ValueError(
                f"Unrecognized OTEL_SPAN_PROCESSOR={config.span_processor!r}; "
                "expected 'simple' or 'batch'."
            )

        resource = Resource.create({"service.name": config.service_name})
        provider = TracerProvider(resource=resource)

        exporter = JSONLFileSpanExporter(config.trace_file_path)
        if config.span_processor == "batch":
            processor_cls = BatchSpanProcessor
        else:
            processor_cls = SimpleSpanProcessor
        provider.add_span_processor(processor_cls(exporter))

        if config.console_export:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)  # best-effort; see docstring above.

        _provider = provider
        _configured = True
        return _provider


def get_tracer(name: str) -> Tracer:
    """Return a tracer from *our own cached* provider (never the ambient
    `opentelemetry.trace.get_tracer()` global lookup).

    This matters because OpenTelemetry's global API only lets
    `set_tracer_provider()` succeed once process-wide -- depending on our own
    cached reference instead sidesteps that entirely and keeps
    `setup_tracing()`'s idempotency independently testable.
    """
    return setup_tracing().get_tracer(name)
