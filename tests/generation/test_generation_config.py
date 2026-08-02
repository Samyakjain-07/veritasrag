from src.generation.config import FaithfulnessConfig, GenerationConfig


def test_generation_config_defaults_load_without_error():
    config = GenerationConfig()

    assert config.generation_max_context_chunks > 0
    assert config.generation_max_chars_per_chunk > 0
    assert config.generation_temperature >= 0.0
    assert "context" in config.generation_system_prompt.lower()
    assert config.generation_no_context_answer


def test_generation_config_env_var_override(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_CONTEXT_CHUNKS", "2")
    monkeypatch.setenv("GENERATION_TEMPERATURE", "0.7")

    config = GenerationConfig()

    assert config.generation_max_context_chunks == 2
    assert config.generation_temperature == 0.7


def test_generation_config_instances_are_independent():
    config_a = GenerationConfig(generation_max_context_chunks=1)
    config_b = GenerationConfig()

    assert config_a.generation_max_context_chunks == 1
    assert config_b.generation_max_context_chunks != 1


def test_faithfulness_config_defaults_load_without_error():
    config = FaithfulnessConfig()

    assert config.faithfulness_nli_model_name
    assert 0.0 <= config.faithfulness_entailment_threshold <= 1.0
    assert config.faithfulness_min_claim_chars > 0


def test_faithfulness_config_env_var_override(monkeypatch):
    monkeypatch.setenv("FAITHFULNESS_ENTAILMENT_THRESHOLD", "0.8")

    config = FaithfulnessConfig()

    assert config.faithfulness_entailment_threshold == 0.8
