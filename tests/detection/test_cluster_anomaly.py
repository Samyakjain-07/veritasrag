import numpy as np
from src.detection.cluster_anomaly import score_cluster_anomaly
from src.detection.config import DetectionConfig


def _make_embeddings(seed: int = 0) -> np.ndarray:
    """30 diffuse points plus a tight 3-point cluster, all L2-normalized."""
    rng = np.random.default_rng(seed)
    diffuse = rng.normal(size=(30, 16))
    tight_center = rng.normal(size=(1, 16))
    tight_cluster = tight_center + rng.normal(scale=0.01, size=(3, 16))

    embeddings = np.vstack([diffuse, tight_cluster])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)


def test_tight_cluster_scores_higher_than_diffuse_background():
    embeddings = _make_embeddings()
    config = DetectionConfig(cluster_dbscan_eps=0.05, cluster_dbscan_min_samples=2)

    result = score_cluster_anomaly(embeddings, config=config)

    tight_cluster_scores = result.normalized_scores[-3:]
    diffuse_scores = result.normalized_scores[:30]

    assert tight_cluster_scores.mean() > diffuse_scores.mean()
    # The 3-point cluster should be identified as its own small DBSCAN cluster.
    assert len(set(result.dbscan_labels[-3:])) == 1
    assert result.dbscan_labels[-3] != -1


def test_scores_are_in_unit_range():
    embeddings = _make_embeddings()

    result = score_cluster_anomaly(embeddings)

    assert (result.normalized_scores >= 0).all()
    assert (result.normalized_scores <= 1).all()


def test_disjoint_reference_embeddings_supported():
    query = _make_embeddings(seed=1)[:5]
    reference = _make_embeddings(seed=2)

    result = score_cluster_anomaly(query, reference_embeddings=reference)

    assert len(result.normalized_scores) == 5
