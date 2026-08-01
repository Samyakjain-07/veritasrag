from src.detection.config import DetectionConfig


def test_defaults_load_without_error():
    config = DetectionConfig()

    assert config.perplexity_model_name == "distilgpt2"
    assert 0.0 <= config.poison_score_threshold <= 1.0
    assert config.cluster_knn_k > 0
    assert config.classifier_n_splits > 1


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_THRESHOLD", "42.5")

    config = DetectionConfig()

    assert config.perplexity_threshold == 42.5


def test_config_instances_are_independent():
    config_a = DetectionConfig(poison_score_threshold=0.9)
    config_b = DetectionConfig()

    assert config_a.poison_score_threshold == 0.9
    assert config_b.poison_score_threshold != 0.9
