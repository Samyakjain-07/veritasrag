"""Lightweight scikit-learn classifier, optionally fused into the detection pipeline.

Trained on chunk embeddings stacked with the normalized perplexity and
cluster-anomaly signals from the other two detectors, fit against Phase 1
manifest labels. Deliberately a scikit-learn model rather than a fine-tuned
HF transformer: only 280 labeled examples (60 positive) are available and
this box is CPU-only, both of which favor a fast, low-variance linear model
over a transformer fine-tune that would be slow to iterate on and prone to
overfitting at this scale.

Evaluated via stratified 5-fold cross-validation rather than a single
train/held-out split: with only 60 positive examples, an 80/20 split leaves
~12 held-out positives, and precision/recall computed over 12 examples is
high-variance (one misclassification swings recall by ~8 points).
`cross_val_predict` gives every one of the 280 chunks an out-of-fold
prediction -- each from a fold that never trained on it, so still
leakage-free -- while using the full manifest for the reported metric. A
*separate* model, fit on the entire manifest, is what actually gets
persisted for production scoring (`pipeline.py`); its own in-sample
predictions are never reported as a metric, only the CV out-of-fold
predictions are.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

# Allow running this file directly (`python -m src.detection.classifier`)
# without requiring the repo to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.detection.config import DEFAULT_CONFIG, DetectionConfig  # noqa: E402

DEFAULT_MODEL_PATH = Path("models/poison_classifier.joblib")
DEFAULT_MANIFEST_PATH = Path("eval/benchmarks/poison_manifest.json")


@dataclass
class ClassifierArtifact:
    """A trained classifier plus the honest, leakage-free metrics it earned."""

    model: Pipeline
    cv_predictions: np.ndarray
    cv_metrics: dict[str, float]
    n_splits: int
    seed: int


def build_features(
    embeddings: np.ndarray, perplexity_norm: np.ndarray, cluster_norm: np.ndarray
) -> np.ndarray:
    """Concatenate chunk embeddings with normalized perplexity/cluster-anomaly scores.

    Normalized (not raw) sub-scores are used so the ~[-1, 1]-scaled embedding
    dimensions aren't drowned out or dominated by raw perplexity's much wider
    numeric range (roughly 12-180+) under the classifier's regularization.
    """
    perplexity_col = np.asarray(perplexity_norm, dtype=np.float64).reshape(-1, 1)
    cluster_col = np.asarray(cluster_norm, dtype=np.float64).reshape(-1, 1)
    return np.hstack([np.asarray(embeddings, dtype=np.float64), perplexity_col, cluster_col])


def _build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced")),
        ]
    )


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    config: DetectionConfig = DEFAULT_CONFIG,
) -> ClassifierArtifact:
    """Fit a classifier and evaluate it via leakage-free stratified cross-validation.

    Returns an artifact whose `.model` is fit on the *entire* `features`/
    `labels` (for production scoring of new chunks) but whose `.cv_metrics`/
    `.cv_predictions` come from out-of-fold cross-validation (for honest
    reporting) -- never mix the two.
    """
    labels = np.asarray(labels, dtype=int)
    cv = StratifiedKFold(
        n_splits=config.classifier_n_splits, shuffle=True, random_state=config.classifier_seed
    )

    cv_predictions = cross_val_predict(
        _build_model(), features, labels, cv=cv, method="predict_proba"
    )[:, 1]

    predicted_labels = (cv_predictions >= config.classifier_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_labels, average="binary", zero_division=0
    )
    cv_metrics = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    model = _build_model()
    model.fit(features, labels)

    return ClassifierArtifact(
        model=model,
        cv_predictions=cv_predictions,
        cv_metrics=cv_metrics,
        n_splits=config.classifier_n_splits,
        seed=config.classifier_seed,
    )


def save_classifier(artifact: ClassifierArtifact, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a trained classifier artifact to disk via joblib.

    Serializes a plain dict of the artifact's fields rather than the
    `ClassifierArtifact` dataclass instance itself: pickling the instance
    directly would record its class as belonging to whichever module happens
    to be `__main__` at training time (e.g. `__main__` when training via
    `python -m src.detection.classifier train`), which then fails to
    unpickle from a different entry point such as `eval/run_eval.py`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": artifact.model,
        "cv_predictions": artifact.cv_predictions,
        "cv_metrics": artifact.cv_metrics,
        "n_splits": artifact.n_splits,
        "seed": artifact.seed,
    }
    joblib.dump(payload, path)
    return path


def load_classifier(path: str | Path = DEFAULT_MODEL_PATH) -> ClassifierArtifact | None:
    """Load a persisted classifier artifact, or None if it hasn't been trained yet.

    The classifier is an optional signal in the fusion pipeline -- callers
    must handle a None return by omitting this signal, not by raising.
    """
    path = Path(path)
    if not path.exists():
        return None
    payload = joblib.load(path)
    return ClassifierArtifact(**payload)


def score(artifact: ClassifierArtifact, features: np.ndarray) -> np.ndarray:
    """Return P(poisoned) for each row of `features` under a trained artifact."""
    return artifact.model.predict_proba(features)[:, 1]


def _train_from_manifest(manifest_path: Path, config: DetectionConfig) -> ClassifierArtifact:
    from src.detection.cluster_anomaly import score_cluster_anomaly
    from src.detection.perplexity import PerplexityScorer, normalize_perplexity
    from src.embeddings.embedder import Embedder

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    texts = [entry["text"] for entry in manifest]
    labels = np.array([int(entry["is_poisoned"]) for entry in manifest])

    embedder = Embedder()
    embeddings = embedder.embed(texts)

    perplexity_scorer = PerplexityScorer(config.perplexity_model_name)
    perplexity_norm = normalize_perplexity(perplexity_scorer.compute_batch(texts), config)

    cluster_result = score_cluster_anomaly(embeddings, config=config)

    features = build_features(embeddings, perplexity_norm, cluster_result.normalized_scores)
    return train_classifier(features, labels, config)


def _cmd_train(args: argparse.Namespace) -> None:
    artifact = _train_from_manifest(args.manifest, DEFAULT_CONFIG)
    output_path = save_classifier(artifact, args.output)

    print(f"Trained classifier saved to {output_path}")
    print(f"{artifact.n_splits}-fold CV metrics (out-of-fold, leakage-free): {artifact.cv_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train the classifier on the Phase 1 manifest and persist it"
    )
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.set_defaults(func=_cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
