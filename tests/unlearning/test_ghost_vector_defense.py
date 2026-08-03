import numpy as np
from src.unlearning.ghost_vector_defense import attempt_reconstruction


def _random_unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_planted_near_duplicate_exceeds_chance():
    rng = np.random.default_rng(0)
    true_vector = _random_unit_vectors(rng, 1, 32)[0]

    survivors = _random_unit_vectors(rng, 15, 32)
    near_duplicate = true_vector + rng.normal(scale=0.01, size=32).astype(np.float32)
    near_duplicate /= np.linalg.norm(near_duplicate)
    survivors = np.vstack([survivors, near_duplicate])

    decoys = _random_unit_vectors(rng, 20, 32)

    result = attempt_reconstruction(true_vector, survivors, decoys)

    assert result.attack_similarity > 0.95
    assert result.exceeds_chance is True


def test_no_duplicate_does_not_exceed_chance():
    rng = np.random.default_rng(1)
    true_vector = _random_unit_vectors(rng, 1, 32)[0]
    survivors = _random_unit_vectors(rng, 15, 32)
    decoys = _random_unit_vectors(rng, 20, 32)

    result = attempt_reconstruction(true_vector, survivors, decoys)

    assert result.exceeds_chance is False


def test_empty_survivors_does_not_crash():
    rng = np.random.default_rng(2)
    true_vector = _random_unit_vectors(rng, 1, 8)[0]
    survivors = np.empty((0, 8), dtype=np.float32)
    decoys = _random_unit_vectors(rng, 5, 8)

    result = attempt_reconstruction(true_vector, survivors, decoys)

    assert result.attack_similarity == 0.0
    assert result.exceeds_chance is False


def test_chance_baseline_uses_percentile_not_mean():
    """Regression test for the max-vs-mean chance-baseline bug: a decoy pool
    with two high-similarity outliers and eighteen near-zero-similarity
    draws produces a low mean but a high p95 threshold. The attack statistic
    sits between the two -- above the mean (so a naive mean-based comparison
    would wrongly call this a successful reconstruction) but below the p95
    threshold (the correct, non-exceeding verdict).
    """
    single_survivor = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    true_vector = np.array([0.9, np.sqrt(1 - 0.9**2), 0.0], dtype=np.float32)

    orthogonal_decoy = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    high_similarity_decoy = np.array([0.95, np.sqrt(1 - 0.95**2), 0.0], dtype=np.float32)
    decoys = np.vstack([np.tile(orthogonal_decoy, (18, 1)), np.tile(high_similarity_decoy, (2, 1))])

    result = attempt_reconstruction(true_vector, single_survivor, decoys)

    assert result.chance_baseline_mean < result.attack_similarity
    assert result.attack_similarity < result.chance_baseline_percentile_threshold
    assert result.exceeds_chance is False
