from src.retrieval.config import PolicyConfig


def test_defaults_load_without_error():
    config = PolicyConfig()

    assert config.retrieval_top_k > 0
    assert config.retrieval_collection_name == "veritasrag_chunks"
    assert config.allow_quarantined_tier is False
    assert config.frequency_hard_block is False
    assert (
        0.0
        < config.poison_downweight_threshold_verified
        < config.poison_hard_drop_threshold_verified
        <= 1.0
    )
    assert (
        0.0
        < config.poison_downweight_threshold_unverified
        < config.poison_hard_drop_threshold_unverified
        <= 1.0
    )


def test_verified_band_is_more_permissive_than_unverified():
    config = PolicyConfig()

    assert (
        config.poison_downweight_threshold_verified > config.poison_downweight_threshold_unverified
    )
    assert config.poison_hard_drop_threshold_verified > config.poison_hard_drop_threshold_unverified


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "8")
    monkeypatch.setenv("POLICY_ALLOW_QUARANTINED_TIER", "true")
    monkeypatch.setenv("FREQUENCY_HARD_BLOCK", "true")

    config = PolicyConfig()

    assert config.retrieval_top_k == 8
    assert config.allow_quarantined_tier is True
    assert config.frequency_hard_block is True


def test_config_instances_are_independent():
    config_a = PolicyConfig(retrieval_top_k=1)
    config_b = PolicyConfig()

    assert config_a.retrieval_top_k == 1
    assert config_b.retrieval_top_k != 1


def test_exit_criteria_thresholds_are_config_driven(monkeypatch):
    monkeypatch.setenv("EXIT_CRITERIA_ASR_REDUCTION_PP_MIN", "40.0")
    monkeypatch.setenv("EXIT_CRITERIA_PRECISION_LOSS_RELATIVE_PCT_MAX", "10.0")

    config = PolicyConfig()

    assert config.exit_criteria_asr_reduction_pp_min == 40.0
    assert config.exit_criteria_precision_loss_relative_pct_max == 10.0
