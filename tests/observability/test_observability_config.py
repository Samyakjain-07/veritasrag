from src.observability.config import ObservabilityConfig


def test_defaults_load_without_error():
    config = ObservabilityConfig()

    assert config.service_name == "veritasrag"
    assert config.trace_file_path == "observability/traces.jsonl"
    assert config.span_processor == "simple"
    assert config.console_export is False


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "custom-service")
    monkeypatch.setenv("OTEL_TRACE_FILE_PATH", "custom/traces.jsonl")
    monkeypatch.setenv("OTEL_SPAN_PROCESSOR", "batch")
    monkeypatch.setenv("OTEL_CONSOLE_EXPORT", "true")

    config = ObservabilityConfig()

    assert config.service_name == "custom-service"
    assert config.trace_file_path == "custom/traces.jsonl"
    assert config.span_processor == "batch"
    assert config.console_export is True


def test_config_instances_are_independent():
    config_a = ObservabilityConfig(service_name="a")
    config_b = ObservabilityConfig()

    assert config_a.service_name == "a"
    assert config_b.service_name != "a"
