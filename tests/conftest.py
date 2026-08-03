"""Session-wide test fixtures.

`_isolate_observability_tracing` redirects the process-wide OTel
`TracerProvider` (a first-caller-wins singleton, see
`src/observability/otel_setup.py::setup_tracing`) to a scratch trace file
*before any test body runs*. Without this, whichever test happens to be the
first of the whole suite to call an instrumented function (`retrieve()`,
`generate()`, `upsert_manifest()`, ...) would silently lock in real writes to
the repo's actual `observability/traces.jsonl` for the rest of the session --
this fixture wins that race instead, matching the existing convention of
never letting tests touch real generated-artifact dirs (e.g. `receipts/`
tests already redirect via `tmp_path`).

Session-scoped and autouse: runs once, after pytest has finished importing
every test module (and everything those modules import) but before the first
test body executes -- exactly the window `setup_tracing()`'s docstring
describes as safe.
"""

from __future__ import annotations

import pytest
from src.observability import otel_setup
from src.observability.config import ObservabilityConfig


@pytest.fixture(scope="session", autouse=True)
def _isolate_observability_tracing(tmp_path_factory: pytest.TempPathFactory) -> None:
    trace_path = tmp_path_factory.mktemp("otel") / "traces.jsonl"
    otel_setup.setup_tracing(ObservabilityConfig(trace_file_path=str(trace_path)))
