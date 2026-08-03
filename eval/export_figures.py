"""Export result figures for the results.md report (Stage 3).

Regenerates two figures from real, already-computed artifacts:

- `perplexity_distribution.png`: raw `distilgpt2` perplexity per chunk
  (`src.detection.perplexity.PerplexityScorer`), split clean vs. poisoned,
  over the full 280-chunk `poison_manifest.json`. Deterministic (no
  sampling/training involved -- a causal-LM forward pass is a pure function
  of its input), so this is a straight re-derivation of the same numbers
  already summarized in `perplexity.py`'s module docstring, not a new eval.
- `detector_roc_det.png`: ROC and DET curves for the classifier signal,
  built from `models/poison_classifier.joblib`'s persisted `cv_predictions`
  -- the leakage-free, 5-fold `cross_val_predict` out-of-fold probabilities
  from Phase 3 (see `src/detection/classifier.py`), not a fresh model fit.

Does NOT attempt an ASR-vs-poison-rate curve: that needs corpora generated
at multiple poison ratios, which don't exist yet (see results.md Stage 1,
Table III gap note).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import det_curve, roc_auc_score, roc_curve  # noqa: E402
from src.detection.perplexity import PerplexityScorer  # noqa: E402

MANIFEST_PATH = Path("eval/benchmarks/poison_manifest.json")
CLASSIFIER_PATH = Path("models/poison_classifier.joblib")
FIGURES_DIR = Path("eval/benchmarks/figures")
RAW_SCORES_PATH = Path("eval/benchmarks/perplexity_raw_scores.json")


def export_perplexity_distribution() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    texts = [entry["text"] for entry in manifest]
    labels = [bool(entry["is_poisoned"]) for entry in manifest]

    print(f"Scoring {len(texts)} chunks with distilgpt2 (CPU, no grad)...")
    scorer = PerplexityScorer()
    raw_scores = scorer.compute_batch(texts)

    RAW_SCORES_PATH.write_text(
        json.dumps(
            {
                "chunk_id": [entry["chunk_id"] for entry in manifest],
                "is_poisoned": labels,
                "perplexity": raw_scores,
            },
            indent=2,
        )
    )

    clean_scores = np.array([s for s, p in zip(raw_scores, labels, strict=True) if not p])
    poison_scores = np.array([s for s, p in zip(raw_scores, labels, strict=True) if p])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(raw_scores) * 1.05, 40)
    ax.hist(
        clean_scores, bins=bins, alpha=0.6, label=f"clean (n={len(clean_scores)})", color="#4C72B0"
    )
    ax.hist(
        poison_scores,
        bins=bins,
        alpha=0.6,
        label=f"poisoned (n={len(poison_scores)})",
        color="#C44E52",
    )
    ax.axvline(60.0, color="black", linestyle="--", linewidth=1, label="detector threshold (60.0)")
    ax.set_xlabel("distilgpt2 perplexity")
    ax.set_ylabel("chunk count")
    ax.set_title("Perplexity distribution: clean vs. poisoned chunks (n=280)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perplexity_distribution.png", dpi=150)
    plt.close(fig)

    stats = {
        "clean_mean": float(clean_scores.mean()),
        "clean_median": float(np.median(clean_scores)),
        "poison_mean": float(poison_scores.mean()),
        "poison_median": float(np.median(poison_scores)),
        "roc_auc": float(roc_auc_score(labels, raw_scores)),
    }
    print("Perplexity stats:", stats)
    return stats


def export_detector_roc_det() -> dict:
    artifact = joblib.load(CLASSIFIER_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text())
    labels = np.array([int(entry["is_poisoned"]) for entry in manifest])
    scores = np.asarray(artifact["cv_predictions"])
    assert len(labels) == len(scores), "manifest/classifier-artifact length mismatch"

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    det_fpr, det_fnr, _ = det_curve(labels, scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, color="#4C72B0", label=f"classifier (CV out-of-fold), AUC={auc:.3f}")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="chance")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC curve (poison detector)")
    ax1.legend(loc="lower right")

    ax2.plot(det_fpr, det_fnr, color="#C44E52")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("False Positive Rate (log)")
    ax2.set_ylabel("False Negative Rate (log)")
    ax2.set_title("DET curve (poison detector)")

    fig.suptitle(
        f"Detector operating curves — classifier signal, 5-fold CV out-of-fold "
        f"(n={len(labels)}, {int(labels.sum())} poisoned)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "detector_roc_det.png", dpi=150)
    plt.close(fig)

    stats = {"roc_auc": auc, "n_thresholds_roc": len(fpr)}
    print("ROC/DET stats:", stats)
    return stats


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    perplexity_stats = export_perplexity_distribution()
    roc_stats = export_detector_roc_det()
    summary = {"perplexity": perplexity_stats, "detector_roc": roc_stats}
    (FIGURES_DIR / "figure_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
