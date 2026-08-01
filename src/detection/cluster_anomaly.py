"""Embedding-space cluster anomaly detection.

PoisonedRAG-style attacks typically inject several adversarial passages that
target the same question (see `build_poison_corpus` in
`eval/synthetic_poison_generator.py`, which generates 3 variants per target
question sharing the same embedded question text). Those variants embed
measurably closer to each other than unrelated corpus passages do (mean
cosine similarity 0.645 for same-target poison pairs vs. ~0.43-0.49 for every
other pair type in the Phase 1 manifest), producing small, dense pockets in
an otherwise diffuse embedding space.

Two signals are combined here, both validated against the real manifest:

- k-NN density (via faiss): mean similarity to a chunk's k nearest
  neighbors. The strongest single cluster-based signal tested (0.367
  precision @ top-60 flagged).
- DBSCAN small-cluster membership: chunks in small, tight DBSCAN clusters
  score as anomalous. Best recall of anything tested (0.633 @ eps=0.25).

Standard sparse-region outlier detectors (`LocalOutlierFactor`,
`IsolationForest`) were also benchmarked and scored *below* the 21.4% base
rate -- they look for the opposite pathology (isolated sparse points), not
dense micro-clusters in a diffuse background. Deliberately not used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from sklearn.cluster import DBSCAN

from src.detection.config import DEFAULT_CONFIG, DetectionConfig


@dataclass(frozen=True)
class ClusterAnomalyResult:
    """Per-chunk raw and normalized cluster-anomaly scores for a scored batch."""

    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    knn_scores: np.ndarray
    dbscan_scores: np.ndarray
    dbscan_labels: np.ndarray


def _knn_density_scores(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int,
    self_referential: bool,
) -> np.ndarray:
    """Mean cosine similarity of each query embedding to its k nearest reference neighbors."""
    dim = reference_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(reference_embeddings.astype(np.float32))

    # Self-referential queries find themselves as their own nearest neighbor
    # (similarity ~1.0) -- search one extra and drop that self-match.
    requested_k = k + 1 if self_referential else k
    search_k = min(requested_k, len(reference_embeddings))
    similarities, _ = index.search(query_embeddings.astype(np.float32), search_k)

    if self_referential and search_k > 1:
        similarities = similarities[:, 1:]

    return similarities.mean(axis=1)


def _dbscan_small_cluster_scores(
    embeddings: np.ndarray, eps: float, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flag chunks belonging to small, tight DBSCAN clusters as anomalous.

    Self-referential only: unlike faiss's index-once/query-many k-NN search,
    DBSCAN has no cheap way to score a new point against an already-fit
    clustering. A disjoint reference/query DBSCAN mode is an explicit
    extension point for Phase 4, not solved here.
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embeddings)
    scores = np.zeros(len(embeddings), dtype=np.float64)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist(), strict=True))
    non_noise_sizes = [size for label, size in cluster_sizes.items() if label != -1]
    max_cluster_size = max(non_noise_sizes, default=1)

    for i, label in enumerate(labels):
        if label == -1:
            # Isolated points are somewhat suspicious but far less so than
            # membership in a small, dense cluster -- fixed low baseline.
            scores[i] = 0.25
        else:
            size = cluster_sizes[label]
            scores[i] = 1.0 - min(size / max_cluster_size, 1.0)

    return scores, labels


def score_cluster_anomaly(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray | None = None,
    config: DetectionConfig = DEFAULT_CONFIG,
) -> ClusterAnomalyResult:
    """Score each row of `query_embeddings` for embedding-space cluster anomaly.

    `reference_embeddings` defaults to `query_embeddings` (self-referential --
    what the Phase 3 eval uses). A disjoint reference corpus is supported for
    the k-NN density sub-signal (e.g. scoring a handful of freshly retrieved
    candidates against a precomputed corpus index), but the DBSCAN sub-signal
    always clusters `query_embeddings` against itself regardless (see
    `_dbscan_small_cluster_scores`).

    Raw scores are a weighted sum of the two sub-signals; normalized scores
    are min-max scaled across this batch, since cluster anomaly is a
    relative-density measure by definition -- unlike perplexity, there is no
    fixed anchor that means the same thing across batches of different size
    or composition. This is a known limitation for callers that score small,
    varying-size batches (e.g. Phase 4's per-query candidate sets).
    """
    self_referential = reference_embeddings is None
    reference = query_embeddings if self_referential else reference_embeddings

    knn_scores = _knn_density_scores(
        query_embeddings, reference, config.cluster_knn_k, self_referential
    )
    dbscan_scores, dbscan_labels = _dbscan_small_cluster_scores(
        query_embeddings, config.cluster_dbscan_eps, config.cluster_dbscan_min_samples
    )

    raw_scores = (
        config.cluster_knn_weight * knn_scores + config.cluster_dbscan_weight * dbscan_scores
    )

    span = raw_scores.max() - raw_scores.min()
    normalized_scores = (
        (raw_scores - raw_scores.min()) / span if span > 0 else np.zeros_like(raw_scores)
    )

    return ClusterAnomalyResult(
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
        knn_scores=knn_scores,
        dbscan_scores=dbscan_scores,
        dbscan_labels=dbscan_labels,
    )
