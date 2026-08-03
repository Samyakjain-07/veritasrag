from src.unlearning.config import UnlearningConfig


def test_defaults_load_without_error():
    config = UnlearningConfig()

    assert config.unlearning_collection_name == "veritasrag_chunks"
    assert config.delete_wait is True
    assert config.compaction_restore_original_config is True
    assert config.ghost_defense_search_k > 0
    assert config.reconstruction_chance_sample_size > 0
    assert 0.0 < config.reconstruction_chance_percentile <= 100.0
    assert config.receipt_dir == "receipts"
    assert config.eval_sample_size > 0


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("UNLEARNING_COLLECTION_NAME", "custom_collection")
    monkeypatch.setenv("RECONSTRUCTION_CHANCE_PERCENTILE", "90.0")
    monkeypatch.setenv("UNLEARNING_RECEIPT_DIR", "custom_receipts")
    monkeypatch.setenv("UNLEARNING_COMPACTION_RESTORE_ORIGINAL_CONFIG", "false")

    config = UnlearningConfig()

    assert config.unlearning_collection_name == "custom_collection"
    assert config.reconstruction_chance_percentile == 90.0
    assert config.receipt_dir == "custom_receipts"
    assert config.compaction_restore_original_config is False


def test_config_instances_are_independent():
    config_a = UnlearningConfig(eval_sample_size=1)
    config_b = UnlearningConfig()

    assert config_a.eval_sample_size == 1
    assert config_b.eval_sample_size != 1
