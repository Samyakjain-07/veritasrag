import numpy as np
from src.detection.classifier import (
    build_features,
    load_classifier,
    save_classifier,
    score,
    train_classifier,
)
from src.detection.config import DetectionConfig


def _synthetic_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """30 rows (15 clean / 15 poisoned), clearly separable by construction."""
    rng = np.random.default_rng(seed)
    n_per_class = 15
    dim = 8

    clean_embeddings = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, dim))
    poison_embeddings = rng.normal(loc=3.0, scale=1.0, size=(n_per_class, dim))
    embeddings = np.vstack([clean_embeddings, poison_embeddings])

    perplexity_norm = np.concatenate(
        [rng.uniform(0, 0.3, n_per_class), rng.uniform(0.7, 1.0, n_per_class)]
    )
    cluster_norm = np.concatenate(
        [rng.uniform(0, 0.3, n_per_class), rng.uniform(0.7, 1.0, n_per_class)]
    )
    labels = np.array([0] * n_per_class + [1] * n_per_class)

    features = build_features(embeddings, perplexity_norm, cluster_norm)
    return features, labels


def test_build_features_shape():
    embeddings = np.zeros((5, 4))
    perplexity_norm = np.zeros(5)
    cluster_norm = np.zeros(5)

    features = build_features(embeddings, perplexity_norm, cluster_norm)

    assert features.shape == (5, 6)


def test_train_classifier_separates_clearly_separable_classes():
    features, labels = _synthetic_dataset()
    config = DetectionConfig(classifier_n_splits=5, classifier_seed=42)

    artifact = train_classifier(features, labels, config)

    assert artifact.cv_metrics["f1"] > 0.7
    assert len(artifact.cv_predictions) == len(labels)


def test_load_classifier_returns_none_for_missing_path(tmp_path):
    result = load_classifier(tmp_path / "does_not_exist.joblib")

    assert result is None


def test_save_and_load_classifier_roundtrip(tmp_path):
    features, labels = _synthetic_dataset()
    artifact = train_classifier(features, labels, DetectionConfig())

    path = save_classifier(artifact, tmp_path / "model.joblib")
    loaded = load_classifier(path)

    assert loaded is not None
    scores_before = score(artifact, features)
    scores_after = score(loaded, features)
    assert np.allclose(scores_before, scores_after)
