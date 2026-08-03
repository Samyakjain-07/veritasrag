import json
import threading

import pytest
from src.observability import otel_setup
from src.observability.config import ObservabilityConfig


@pytest.fixture(autouse=True)
def _reset_otel_state(monkeypatch):
    """Give every test in this file a fresh, unconfigured provider singleton.

    `monkeypatch.setattr` reverts these two module globals back to whatever
    the session-wide fixture (`tests/conftest.py`) already established once
    this test function returns -- so re-exercising `setup_tracing()` here
    never leaks a differently-configured provider into other test files.
    """
    monkeypatch.setattr(otel_setup, "_configured", False)
    monkeypatch.setattr(otel_setup, "_provider", None)


def test_jsonl_exporter_writes_one_line_per_span(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    otel_setup.setup_tracing(ObservabilityConfig(trace_file_path=str(trace_path)))
    tracer = otel_setup.get_tracer("test")

    with tracer.start_as_current_span("test.span") as span:
        span.set_attribute("foo", 1)

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert len(record["trace_id"]) == 32
    assert all(c in "0123456789abcdef" for c in record["trace_id"])
    assert len(record["span_id"]) == 16
    assert record["parent_span_id"] is None
    assert record["name"] == "test.span"
    assert record["status_code"] == "UNSET"
    assert record["service_name"] == "veritasrag"
    assert record["attributes"]["foo"] == 1


def test_parent_child_span_linkage(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    otel_setup.setup_tracing(ObservabilityConfig(trace_file_path=str(trace_path)))
    tracer = otel_setup.get_tracer("test")

    with tracer.start_as_current_span("parent"):
        with tracer.start_as_current_span("child"):
            pass

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    child = next(r for r in records if r["name"] == "child")
    parent = next(r for r in records if r["name"] == "parent")

    assert child["parent_span_id"] == parent["span_id"]
    assert child["trace_id"] == parent["trace_id"]


def test_concurrent_exports_do_not_interleave_or_corrupt(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    otel_setup.setup_tracing(ObservabilityConfig(trace_file_path=str(trace_path)))
    tracer = otel_setup.get_tracer("test")

    spans_per_thread = 20
    thread_count = 8

    def make_spans() -> None:
        for i in range(spans_per_thread):
            with tracer.start_as_current_span(f"span-{i}"):
                pass

    threads = [threading.Thread(target=make_spans) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == thread_count * spans_per_thread
    for line in lines:
        json.loads(line)  # raises if a write interleaved/corrupted a line


def test_setup_tracing_is_idempotent(tmp_path):
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"

    provider_a = otel_setup.setup_tracing(ObservabilityConfig(trace_file_path=str(path_a)))
    provider_b = otel_setup.setup_tracing(ObservabilityConfig(trace_file_path=str(path_b)))

    assert provider_a is provider_b

    tracer = otel_setup.get_tracer("test")
    with tracer.start_as_current_span("after-second-config"):
        pass

    assert path_a.exists()
    assert path_a.read_text(encoding="utf-8").strip() != ""
    assert not path_b.exists()


def test_get_tracer_returns_working_tracer(tmp_path):
    otel_setup.setup_tracing(ObservabilityConfig(trace_file_path=str(tmp_path / "t.jsonl")))
    tracer = otel_setup.get_tracer("x")

    assert hasattr(tracer, "start_as_current_span")
    with tracer.start_as_current_span("s"):
        pass


def test_setup_tracing_rejects_unknown_span_processor(tmp_path):
    config = ObservabilityConfig(trace_file_path=str(tmp_path / "t.jsonl"), span_processor="bogus")

    with pytest.raises(ValueError, match="Unrecognized OTEL_SPAN_PROCESSOR"):
        otel_setup.setup_tracing(config)
