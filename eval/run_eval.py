"""Phase 3 detection eval: scores the Phase 1 manifest with every detector and
the LLM-as-judge baseline, and reports precision/recall/F1 for each.

Usage:
    python eval/run_eval.py --stage detection
    python eval/run_eval.py --stage detection --skip-judge          # no API key needed
    python eval/run_eval.py --stage detection --judge-sample-size 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running this file directly (`python eval/run_eval.py`) without
# requiring the repo to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402
from src.detection.classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    build_features,
    load_classifier,
    save_classifier,
    train_classifier,
)
from src.detection.cluster_anomaly import score_cluster_anomaly  # noqa: E402
from src.detection.config import DEFAULT_CONFIG, DetectionConfig  # noqa: E402
from src.detection.llm_judge import judge_chunks  # noqa: E402
from src.detection.perplexity import PerplexityScorer, normalize_perplexity  # noqa: E402
from src.detection.pipeline import fuse_scores  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402

DEFAULT_MANIFEST_PATH = Path("eval/benchmarks/poison_manifest.json")
DEFAULT_OUTPUT_PATH = Path("eval/benchmarks/detection_eval_results.json")
DEFAULT_SEED = 42


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _print_report(rows: dict[str, dict]) -> None:
    header = f"{'method':<24}{'precision':>10}{'recall':>10}{'f1':>10}{'n':>8}"
    print(header)
    print("-" * len(header))
    for name, metrics in rows.items():
        n = metrics.get("n", "")
        print(
            f"{name:<24}{metrics['precision']:>10.3f}{metrics['recall']:>10.3f}"
            f"{metrics['f1']:>10.3f}{n:>8}"
        )


def run_detection_eval(
    manifest_path: Path,
    output_path: Path,
    classifier_path: Path,
    skip_judge: bool,
    judge_sample_size: int | None,
    seed: int,
    config: DetectionConfig = DEFAULT_CONFIG,
) -> dict:
    """Run every Phase 3 detector plus the LLM-judge baseline against the manifest.

    Returns the full results dict that also gets written to `output_path`.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    texts = [entry["text"] for entry in manifest]
    chunk_ids = [entry["chunk_id"] for entry in manifest]
    labels = np.array([int(entry["is_poisoned"]) for entry in manifest])

    print(
        f"Loaded {len(manifest)} chunks from {manifest_path} "
        f"({int(labels.sum())} poisoned, {len(labels) - int(labels.sum())} clean)"
    )

    embedder = Embedder()
    embeddings = embedder.embed(texts)

    print("Scoring perplexity...")
    perplexity_scorer = PerplexityScorer(config.perplexity_model_name)
    perplexity_raw = np.array(perplexity_scorer.compute_batch(texts))
    perplexity_norm = normalize_perplexity(perplexity_raw, config)

    print("Scoring cluster anomaly...")
    cluster_result = score_cluster_anomaly(embeddings, config=config)

    print("Loading/training classifier...")
    classifier_artifact = load_classifier(classifier_path)
    if classifier_artifact is None:
        features = build_features(embeddings, perplexity_norm, cluster_result.normalized_scores)
        classifier_artifact = train_classifier(features, labels, config)
        save_classifier(classifier_artifact, classifier_path)
        print(f"  trained fresh classifier, saved to {classifier_path}")
    else:
        print(f"  loaded existing classifier from {classifier_path}")
    classifier_cv_scores = classifier_artifact.cv_predictions

    # Fused pipeline: uses the classifier's leakage-free CV out-of-fold scores
    # (not a live model's in-sample prediction) so this report row isn't
    # inflated by the classifier having trained on the very rows being scored.
    poison_scores = np.array(
        [
            fuse_scores(
                float(perplexity_norm[i]),
                float(cluster_result.normalized_scores[i]),
                float(classifier_cv_scores[i]),
                config,
            )
            for i in range(len(texts))
        ]
    )

    report: dict[str, dict] = {}
    report["perplexity"] = {
        **_binary_metrics(labels, (perplexity_raw >= config.perplexity_threshold).astype(int)),
        "n": len(labels),
    }
    cluster_flags = (cluster_result.normalized_scores >= config.cluster_anomaly_threshold).astype(
        int
    )
    report["cluster_anomaly"] = {**_binary_metrics(labels, cluster_flags), "n": len(labels)}

    classifier_flags = (classifier_cv_scores >= config.classifier_threshold).astype(int)
    report["classifier_cv"] = {**_binary_metrics(labels, classifier_flags), "n": len(labels)}
    report["fused_pipeline"] = {
        **_binary_metrics(labels, (poison_scores >= config.poison_score_threshold).astype(int)),
        "n": len(labels),
    }

    if skip_judge:
        print("Skipping LLM-judge baseline (--skip-judge).")
    else:
        if judge_sample_size is not None and judge_sample_size < len(manifest):
            rng = random.Random(seed)
            judge_indices = sorted(rng.sample(range(len(manifest)), judge_sample_size))
        else:
            judge_indices = list(range(len(manifest)))

        print(
            f"Running LLM-judge over {len(judge_indices)} chunks "
            f"(model={config.openai_judge_model})..."
        )
        judge_texts = [texts[i] for i in judge_indices]
        judge_chunk_ids = [chunk_ids[i] for i in judge_indices]
        judge_results = judge_chunks(judge_texts, judge_chunk_ids, config)

        judge_labels = labels[judge_indices]
        judge_predictions = np.array([int(r.flagged) for r in judge_results])
        report["llm_judge"] = {
            **_binary_metrics(judge_labels, judge_predictions),
            "n": len(judge_indices),
        }

    print()
    _print_report(report)

    results = {
        "manifest_path": str(manifest_path),
        "n_chunks": len(manifest),
        "n_poisoned": int(labels.sum()),
        "config": asdict(config),
        "metrics": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["detection"], default="detection")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--classifier-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--judge-sample-size",
        type=int,
        default=None,
        help="Judge only a seeded random subsample of this many chunks instead of the full set.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_detection_eval(
        manifest_path=args.manifest,
        output_path=args.output,
        classifier_path=args.classifier_path,
        skip_judge=args.skip_judge,
        judge_sample_size=args.judge_sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
