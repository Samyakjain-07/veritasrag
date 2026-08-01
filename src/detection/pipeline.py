"""Fuses the perplexity, cluster-anomaly, and (optional) classifier signals
into a single per-chunk `poison_score`.

The LLM-as-judge baseline (`src/detection/llm_judge.py`) is deliberately
*not* one of the fused inputs here -- per PLAN.md's own exit-criteria
wording ("...report comparing your combined detector *against* the
LLM-as-judge baseline"), it's a standalone comparison baseline, not a fusion
signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.detection.classifier import (
    DEFAULT_MODEL_PATH,
    ClassifierArtifact,
    build_features,
    load_classifier,
)
from src.detection.classifier import score as classifier_score_fn
from src.detection.cluster_anomaly import score_cluster_anomaly
from src.detection.config import DEFAULT_CONFIG, DetectionConfig
from src.detection.perplexity import PerplexityScorer, normalize_perplexity
from src.embeddings.embedder import Embedder


@dataclass(frozen=True)
class ChunkScore:
    """Per-chunk raw/normalized sub-scores and the final fused poison_score."""

    chunk_id: str | None
    text: str
    perplexity_raw: float
    perplexity_norm: float
    cluster_anomaly_raw: float
    cluster_anomaly_norm: float
    classifier_score: float | None
    poison_score: float


def fuse_scores(
    perplexity_norm: float,
    cluster_anomaly_norm: float,
    classifier_score: float | None,
    config: DetectionConfig = DEFAULT_CONFIG,
) -> float:
    """Weighted-sum fusion of the three detection signals into one poison_score.

    Renormalizes weights across whichever signals are actually present, so a
    missing (untrained) classifier doesn't silently deflate the fused score --
    `classifier_score=None` redistributes its configured weight across
    perplexity and cluster-anomaly instead of counting as zero.

    Reused both by `DetectionPipeline.score_chunks` (with its own live
    classifier prediction) and directly by `eval/run_eval.py` (with the
    classifier's cross-validated out-of-fold scores instead, to keep the
    fused-pipeline eval leakage-free) so the fusion math itself is never
    duplicated.
    """
    weights = {
        "perplexity": config.fusion_perplexity_weight,
        "cluster": config.fusion_cluster_weight,
    }
    scores = {
        "perplexity": perplexity_norm,
        "cluster": cluster_anomaly_norm,
    }
    if classifier_score is not None:
        weights["classifier"] = config.fusion_classifier_weight
        scores["classifier"] = classifier_score

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(scores[key] * weights[key] for key in weights) / total_weight


class DetectionPipeline:
    """Loads the embedder, perplexity scorer, and (if trained) classifier once,
    then fuses their per-chunk signals into a single `poison_score` per call."""

    def __init__(
        self,
        config: DetectionConfig = DEFAULT_CONFIG,
        embedder: Embedder | None = None,
        classifier_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.config = config
        self.embedder = embedder or Embedder()
        self.perplexity_scorer = PerplexityScorer(config.perplexity_model_name)
        self.classifier_artifact: ClassifierArtifact | None = load_classifier(classifier_path)

    def score_chunks(
        self,
        texts: list[str],
        chunk_ids: list[str] | None = None,
        reference_texts: list[str] | None = None,
    ) -> list[ChunkScore]:
        """Score a batch of chunks, fusing per-chunk signals into `poison_score`.

        `reference_texts` defaults to `texts` itself (self-referential cluster
        anomaly, matching the Phase 3 eval); passing a disjoint reference
        corpus is forwarded to `score_cluster_anomaly` for future per-query
        use against a precomputed corpus index.
        """
        if chunk_ids is None:
            chunk_ids = [None] * len(texts)

        embeddings = self.embedder.embed(texts)
        reference_embeddings = (
            self.embedder.embed(reference_texts) if reference_texts is not None else None
        )

        perplexity_raw = self.perplexity_scorer.compute_batch(texts)
        perplexity_norm = normalize_perplexity(perplexity_raw, self.config)

        cluster_result = score_cluster_anomaly(
            embeddings, reference_embeddings=reference_embeddings, config=self.config
        )

        if self.classifier_artifact is not None:
            features = build_features(embeddings, perplexity_norm, cluster_result.normalized_scores)
            classifier_scores = classifier_score_fn(self.classifier_artifact, features)
        else:
            classifier_scores = [None] * len(texts)

        results = []
        for i, text in enumerate(texts):
            classifier_score_i = classifier_scores[i]
            poison_score = fuse_scores(
                float(perplexity_norm[i]),
                float(cluster_result.normalized_scores[i]),
                None if classifier_score_i is None else float(classifier_score_i),
                self.config,
            )
            results.append(
                ChunkScore(
                    chunk_id=chunk_ids[i],
                    text=text,
                    perplexity_raw=float(perplexity_raw[i]),
                    perplexity_norm=float(perplexity_norm[i]),
                    cluster_anomaly_raw=float(cluster_result.raw_scores[i]),
                    cluster_anomaly_norm=float(cluster_result.normalized_scores[i]),
                    classifier_score=(
                        None if classifier_score_i is None else float(classifier_score_i)
                    ),
                    poison_score=poison_score,
                )
            )
        return results

    def flag(self, chunk_score: ChunkScore) -> bool:
        """Threshold a chunk's fused poison_score against the configured cutoff."""
        return chunk_score.poison_score >= self.config.poison_score_threshold
