import json
from pathlib import Path

import pytest
from eval.run_eval import (
    DEFAULT_MANIFEST_PATH,
    _attack_succeeded,
    _chunk_to_question_map,
    _percentile,
    _precision_at_k,
    build_query_sets,
)


def _load_real_manifest() -> list[dict]:
    return json.loads(Path(DEFAULT_MANIFEST_PATH).read_text(encoding="utf-8"))


def test_build_query_sets_matches_verified_manifest_split():
    """Verified directly against the real manifest: 220 clean chunks across
    122 distinct rows, 60 poison chunks across 20 distinct target rows, 16 of
    which overlap with a clean-chunk row -> 106 precision queries."""
    manifest = _load_real_manifest()

    asr_queries, precision_queries, n_overlap_excluded = build_query_sets(manifest)

    assert len(asr_queries) == 20
    assert len(precision_queries) == 106
    assert n_overlap_excluded == 16


def test_build_query_sets_asr_queries_carry_poison_ids_only():
    manifest = _load_real_manifest()
    asr_queries, _, _ = build_query_sets(manifest)

    for query in asr_queries:
        assert query.question
        assert len(query.poison_chunk_ids) > 0
        assert query.relevant_chunk_ids == frozenset()


def test_build_query_sets_precision_queries_carry_relevant_ids_only():
    manifest = _load_real_manifest()
    _, precision_queries, _ = build_query_sets(manifest)

    for query in precision_queries:
        assert query.question
        assert len(query.relevant_chunk_ids) > 0
        assert query.poison_chunk_ids == frozenset()


def test_build_query_sets_overlap_excluded_from_precision_only():
    manifest = _load_real_manifest()
    asr_queries, precision_queries, n_overlap_excluded = build_query_sets(manifest)

    asr_rows = {query.row_idx for query in asr_queries}
    precision_rows = {query.row_idx for query in precision_queries}

    assert asr_rows.isdisjoint(precision_rows)
    assert n_overlap_excluded == 16


def test_chunk_to_question_map_covers_every_chunk_in_both_query_sets():
    manifest = _load_real_manifest()
    asr_queries, precision_queries, _ = build_query_sets(manifest)

    mapping = _chunk_to_question_map(asr_queries, precision_queries)

    expected_chunk_ids = set().union(
        *(q.poison_chunk_ids for q in asr_queries),
        *(q.relevant_chunk_ids for q in precision_queries),
    )
    assert set(mapping) == expected_chunk_ids


def test_chunk_to_question_map_poison_chunk_maps_to_target_question():
    manifest = _load_real_manifest()
    asr_queries, precision_queries, _ = build_query_sets(manifest)
    mapping = _chunk_to_question_map(asr_queries, precision_queries)

    query = asr_queries[0]
    chunk_id = next(iter(query.poison_chunk_ids))

    assert mapping[chunk_id] == query.question


def test_attack_succeeded_true_when_poison_chunk_present():
    assert _attack_succeeded(["a", "poison-1", "b"], frozenset({"poison-1", "poison-2"})) is True


def test_attack_succeeded_false_when_no_poison_chunk_present():
    assert _attack_succeeded(["a", "b"], frozenset({"poison-1"})) is False


def test_attack_succeeded_false_on_empty_result():
    assert _attack_succeeded([], frozenset({"poison-1"})) is False


def test_precision_at_k_computes_fraction_relevant():
    assert _precision_at_k(["a", "b", "c"], frozenset({"a", "c"})) == pytest.approx(2 / 3)


def test_precision_at_k_empty_result_is_zero_not_vacuous_one():
    assert _precision_at_k([], frozenset({"a"})) == 0.0


def test_precision_at_k_no_relevant_matches_is_zero():
    assert _precision_at_k(["x", "y"], frozenset({"a"})) == 0.0


def test_precision_at_k_full_overlap_is_one():
    assert _precision_at_k(["a", "b"], frozenset({"a", "b", "c"})) == 1.0


def test_percentile_empty_list_is_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_p50_of_ordered_values():
    assert _percentile([10.0, 20.0, 30.0], 50) == pytest.approx(20.0)


def test_percentile_p95_is_close_to_max_for_small_sample():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]

    assert _percentile(values, 95) >= _percentile(values, 50)
