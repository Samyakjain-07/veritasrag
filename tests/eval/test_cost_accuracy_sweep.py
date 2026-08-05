import pytest
from eval.cost_accuracy_sweep import (
    FULL_DEFENSE_LABEL,
    UNDEFENDED_LABEL,
    _is_dominated,
    _latency_stats,
    _sample_queries,
    build_layer_configs,
)
from eval.run_eval import RetrievalQuery
from src.retrieval.config import PolicyConfig


def _queries(n: int) -> list[RetrievalQuery]:
    return [RetrievalQuery(question=f"q{i}", row_idx=i) for i in range(n)]


def test_sample_queries_returns_all_when_sample_size_is_none():
    queries = _queries(5)

    assert _sample_queries(queries, None, seed=42) == queries


def test_sample_queries_returns_all_when_sample_size_exceeds_length():
    queries = _queries(5)

    assert _sample_queries(queries, 10, seed=42) == queries


def test_sample_queries_returns_requested_subset_size():
    queries = _queries(20)

    sampled = _sample_queries(queries, 5, seed=42)

    assert len(sampled) == 5
    assert set(sampled).issubset(set(queries))


def test_sample_queries_is_deterministic_for_a_fixed_seed():
    queries = _queries(20)

    first = _sample_queries(queries, 5, seed=7)
    second = _sample_queries(queries, 5, seed=7)

    assert first == second


def test_sample_queries_differs_across_seeds_typically():
    queries = _queries(30)

    a = _sample_queries(queries, 5, seed=1)
    b = _sample_queries(queries, 5, seed=2)

    assert a != b


def test_latency_stats_empty_list_is_all_zero():
    stats = _latency_stats([])

    assert stats == {"mean": 0.0, "p50": 0.0, "p95": 0.0}


def test_latency_stats_computes_mean_and_percentiles():
    stats = _latency_stats([10.0, 20.0, 30.0])

    assert stats["mean"] == pytest.approx(20.0)
    assert stats["p50"] == pytest.approx(20.0)
    assert stats["p95"] >= stats["p50"]


def _row(latency: float, asr: float, precision: float) -> dict:
    return {"latency_ms": {"mean": latency}, "asr": asr, "precision_at_k": precision}


def test_is_dominated_true_when_another_row_strictly_better_on_all_axes():
    cheap_and_accurate = _row(latency=10.0, asr=0.0, precision=1.0)
    expensive_and_inaccurate = _row(latency=100.0, asr=0.5, precision=0.5)
    rows = [cheap_and_accurate, expensive_and_inaccurate]

    assert _is_dominated(expensive_and_inaccurate, rows) is True
    assert _is_dominated(cheap_and_accurate, rows) is False


def test_is_dominated_false_when_rows_trade_off_cost_for_accuracy():
    cheap_but_weak = _row(latency=10.0, asr=0.5, precision=0.3)
    costly_but_strong = _row(latency=100.0, asr=0.0, precision=0.9)
    rows = [cheap_but_weak, costly_but_strong]

    assert _is_dominated(cheap_but_weak, rows) is False
    assert _is_dominated(costly_but_strong, rows) is False


def test_is_dominated_false_for_identical_single_row():
    row = _row(latency=10.0, asr=0.1, precision=0.5)

    assert _is_dominated(row, [row]) is False


def test_is_dominated_false_when_values_tie_exactly():
    row_a = _row(latency=10.0, asr=0.1, precision=0.5)
    row_b = _row(latency=10.0, asr=0.1, precision=0.5)

    assert _is_dominated(row_a, [row_a, row_b]) is False


def test_build_layer_configs_includes_full_defense_with_every_toggle_enabled():
    layers = build_layer_configs(PolicyConfig())
    full_defense = next(layer for layer in layers if layer.name == FULL_DEFENSE_LABEL)

    assert full_defense.policy_config.enable_detection_scoring is True
    assert full_defense.policy_config.enable_tier_gating is True
    assert full_defense.policy_config.enable_frequency_tracking is True
    assert full_defense.run_faithfulness_check is True


def test_build_layer_configs_each_ablation_disables_exactly_one_toggle():
    base = PolicyConfig()
    layers = {layer.name: layer for layer in build_layer_configs(base)}

    assert layers["no_detection_scoring"].policy_config.enable_detection_scoring is False
    assert layers["no_detection_scoring"].policy_config.enable_tier_gating is True
    assert layers["no_detection_scoring"].policy_config.enable_frequency_tracking is True

    assert layers["no_tier_gating"].policy_config.enable_tier_gating is False
    assert layers["no_tier_gating"].policy_config.enable_detection_scoring is True

    assert layers["no_frequency_tracking"].policy_config.enable_frequency_tracking is False
    assert layers["no_frequency_tracking"].policy_config.enable_detection_scoring is True

    assert layers["no_faithfulness_check"].run_faithfulness_check is False
    assert layers["no_faithfulness_check"].policy_config == base


def test_build_layer_configs_names_are_unique():
    names = [layer.name for layer in build_layer_configs(PolicyConfig())]

    assert len(names) == len(set(names))
    assert UNDEFENDED_LABEL not in names  # undefended is handled separately, not a LayerConfig
