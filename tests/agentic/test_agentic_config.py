from src.agentic.config import AgenticConfig


def test_agentic_config_defaults_load_without_error():
    config = AgenticConfig()

    assert config.agentic_max_subqueries >= 2
    assert config.agentic_planner_temperature >= 0.0
    assert config.agentic_min_subquery_chars > 0


def test_agentic_config_env_var_override(monkeypatch):
    monkeypatch.setenv("AGENTIC_MAX_SUBQUERIES", "5")
    monkeypatch.setenv("AGENTIC_PLANNER_TEMPERATURE", "0.3")

    config = AgenticConfig()

    assert config.agentic_max_subqueries == 5
    assert config.agentic_planner_temperature == 0.3


def test_agentic_config_instances_are_independent():
    config_a = AgenticConfig(agentic_max_subqueries=1)
    config_b = AgenticConfig()

    assert config_a.agentic_max_subqueries == 1
    assert config_b.agentic_max_subqueries != 1
