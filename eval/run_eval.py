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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

# Allow running this file directly (`python eval/run_eval.py`) without
# requiring the repo to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from eval.synthetic_poison_generator import DEFAULT_DATASET as NQ_DATASET  # noqa: E402
from eval.synthetic_poison_generator import DEFAULT_NUM_QUESTIONS as NQ_NUM_QUESTIONS  # noqa: E402
from eval.synthetic_poison_generator import DEFAULT_SEED as NQ_SEED  # noqa: E402
from eval.synthetic_poison_generator import DEFAULT_SPLIT as NQ_SPLIT  # noqa: E402
from eval.synthetic_poison_generator import load_nq_subset  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402
from src.agentic.config import DEFAULT_CONFIG as DEFAULT_AGENTIC_CONFIG  # noqa: E402
from src.agentic.config import AgenticConfig  # noqa: E402
from src.agentic.graph import merge_sub_results  # noqa: E402
from src.agentic.planner import plan_subqueries  # noqa: E402
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
from src.generation.config import (  # noqa: E402
    DEFAULT_FAITHFULNESS_CONFIG,  # noqa: E402
    FaithfulnessConfig,
    GenerationConfig,
)
from src.generation.config import DEFAULT_GENERATION_CONFIG as DEFAULT_GEN_CONFIG  # noqa: E402
from src.generation.faithfulness import FaithfulnessChecker  # noqa: E402
from src.generation.generate import GenerationResult, generate  # noqa: E402
from src.generation.llm_client import get_llm  # noqa: E402
from src.ingestion.provenance import ProvenanceSigner  # noqa: E402
from src.retrieval.config import DEFAULT_CONFIG as DEFAULT_POLICY_CONFIG  # noqa: E402
from src.retrieval.config import PolicyConfig  # noqa: E402
from src.retrieval.policy_engine import FrequencyTracker, retrieve  # noqa: E402
from src.unlearning.config import DEFAULT_CONFIG as DEFAULT_UNLEARNING_CONFIG  # noqa: E402
from src.unlearning.config import UnlearningConfig  # noqa: E402
from src.unlearning.delete import delete_with_receipt  # noqa: E402
from src.vectorstore.qdrant_client import get_client, search, upsert_manifest  # noqa: E402

DEFAULT_MANIFEST_PATH = Path("eval/benchmarks/poison_manifest.json")
DEFAULT_OUTPUT_PATH = Path("eval/benchmarks/detection_eval_results.json")
DEFAULT_RETRIEVAL_OUTPUT_PATH = Path("eval/benchmarks/retrieval_eval_results.json")
DEFAULT_GENERATION_OUTPUT_PATH = Path("eval/benchmarks/generation_eval_results.json")
DEFAULT_AGENTIC_OUTPUT_PATH = Path("eval/benchmarks/agentic_eval_results.json")
DEFAULT_FULL_REPORT_OUTPUT_PATH = Path("eval/benchmarks/full_report_results.json")
DEFAULT_UNLEARNING_OUTPUT_PATH = Path("eval/benchmarks/unlearning_eval_results.json")
UNLEARNING_EVAL_SCRATCH_COLLECTION = "veritasrag_unlearning_eval_scratch"
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


@dataclass(frozen=True)
class RetrievalQuery:
    """A single eval query: either an attack (target_question of a poisoned
    chunk) or a legitimate lookup (a reconstructed NQ question)."""

    question: str
    row_idx: int
    relevant_chunk_ids: frozenset[str] = frozenset()
    poison_chunk_ids: frozenset[str] = frozenset()


def build_query_sets(
    manifest: list[dict],
) -> tuple[list[RetrievalQuery], list[RetrievalQuery], int]:
    """Reconstruct the ASR and retrieval-precision query sets from a Phase 1 manifest.

    Clean chunks carry no `target_question`, only `source_doc`
    (`f"nq_open::{row_idx}"`), so the underlying NQ sample is re-loaded --
    pinned to `synthetic_poison_generator`'s own seed/params, *not* this
    eval's own `--seed` flag, since it must reproduce the exact manifest
    provenance -- to recover `row_idx -> question` for clean chunks.

    Returns `(asr_queries, precision_queries, n_overlap_excluded)`. Poison
    target rows that also produced a clean chunk are excluded from the
    precision set only (so "precision loss from gating" isn't confounded
    with "the attack itself suppressing a legitimate chunk"); the ASR set
    keeps every target, since shrinking an already-small attack sample would
    cost statistical power for no benefit -- the confound only threatens the
    precision measurement.
    """
    nq_rows = load_nq_subset(NQ_NUM_QUESTIONS, NQ_SEED, dataset_name=NQ_DATASET, split=NQ_SPLIT)
    row_to_question = {row["row_idx"]: row["question"] for row in nq_rows}

    clean_chunks_by_row: dict[int, list[str]] = {}
    poison_chunks_by_row: dict[int, list[str]] = {}
    target_question_by_row: dict[int, str] = {}
    for entry in manifest:
        row_idx = int(entry["source_doc"].split("::")[1])
        if entry["is_poisoned"]:
            poison_chunks_by_row.setdefault(row_idx, []).append(entry["chunk_id"])
            target_question_by_row[row_idx] = entry["target_question"]
        else:
            clean_chunks_by_row.setdefault(row_idx, []).append(entry["chunk_id"])

    overlap_rows = set(clean_chunks_by_row) & set(poison_chunks_by_row)

    asr_queries = [
        RetrievalQuery(
            question=target_question_by_row[row_idx],
            row_idx=row_idx,
            poison_chunk_ids=frozenset(poison_chunks_by_row[row_idx]),
        )
        for row_idx in sorted(poison_chunks_by_row)
    ]
    precision_queries = [
        RetrievalQuery(
            question=row_to_question[row_idx],
            row_idx=row_idx,
            relevant_chunk_ids=frozenset(clean_chunks_by_row[row_idx]),
        )
        for row_idx in sorted(clean_chunks_by_row)
        if row_idx not in overlap_rows
    ]
    return asr_queries, precision_queries, len(overlap_rows)


def _chunk_to_question_map(
    asr_queries: list[RetrievalQuery], precision_queries: list[RetrievalQuery]
) -> dict[str, str]:
    """Flatten both query sets into chunk_id -> its associated question, for
    run_unlearning_eval's before/after removal-precision@k check -- a deleted
    chunk just needs *a* query that used to retrieve it, poison or clean.
    """
    mapping: dict[str, str] = {}
    for query in asr_queries:
        for chunk_id in query.poison_chunk_ids:
            mapping[chunk_id] = query.question
    for query in precision_queries:
        for chunk_id in query.relevant_chunk_ids:
            mapping[chunk_id] = query.question
    return mapping


def _attack_succeeded(chunk_ids: list[str], poison_chunk_ids: frozenset[str]) -> bool:
    """True iff a poisoned chunk targeting this query made it into the result set."""
    return any(chunk_id in poison_chunk_ids for chunk_id in chunk_ids)


def _precision_at_k(chunk_ids: list[str], relevant_chunk_ids: frozenset[str]) -> float:
    """Fraction of the returned chunks that are actually relevant.

    An empty result set scores 0.0, not an excluded/vacuous 1.0 -- a defense
    eval shouldn't reward a policy that over-aggressively returns nothing.
    """
    if not chunk_ids:
        return 0.0
    hits = sum(1 for chunk_id in chunk_ids if chunk_id in relevant_chunk_ids)
    return hits / len(chunk_ids)


def _generation_attack_succeeded(
    generation: GenerationResult, poison_chunk_ids: frozenset[str]
) -> bool:
    """True iff a generated claim was NLI-entailed by a poisoned chunk -- the
    generation-level bar is higher than `_attack_succeeded`'s retrieval-level
    one: surviving retrieval gating isn't enough, the poison must actually
    ground part of the final answer. Shared by `run_generation_eval` (Phase
    5) and `run_agentic_eval` (Phase 8) so this check is never duplicated.
    """
    return bool(
        generation.faithfulness
        and any(
            claim.grounded and claim.best_chunk_id in poison_chunk_ids
            for claim in generation.faithfulness.claims
        )
    )


def _undefended_chunk_ids(
    question: str, embedder: Embedder, client, config: PolicyConfig
) -> list[str]:
    """Raw top-k similarity search: no tier filter, no gating -- the undefended baseline."""
    query_vector = embedder.embed_one(question).tolist()
    hits = search(
        client,
        query_vector,
        collection_name=config.retrieval_collection_name,
        top_k=config.retrieval_top_k,
        trust_tiers=None,
    )
    return [hit.payload["chunk_id"] for hit in hits]


def _print_retrieval_report(metrics: dict) -> None:
    header = f"{'':<12}{'asr':>10}{'precision@k':>14}"
    print(header)
    print("-" * len(header))
    for name in ("defended", "undefended"):
        if name not in metrics:
            continue
        row = metrics[name]
        print(f"{name:<12}{row['asr']:>10.3f}{row['precision_at_k']:>14.3f}")
    if "deltas" in metrics:
        deltas = metrics["deltas"]
        print()
        print(f"ASR reduction (pp):        {deltas['asr_reduction_pp']:+.1f}")
        print(f"Precision loss, abs (pp):  {deltas['precision_loss_pp']:+.1f}")
        print(f"Precision loss, rel (%):   {deltas['precision_loss_relative_pct']:+.1f}")


def run_retrieval_eval(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_RETRIEVAL_OUTPUT_PATH,
    compare_baseline: bool = False,
    config: PolicyConfig = DEFAULT_POLICY_CONFIG,
) -> dict:
    """Run the Phase 4 retrieval-policy eval: defended ASR/precision@k, and --
    if `compare_baseline` -- an undefended-baseline comparison against the
    (config-driven) exit criteria: >= `exit_criteria_asr_reduction_pp_min`
    percentage points of ASR reduction at < `exit_criteria_precision_loss_
    relative_pct_max` percent relative retrieval-precision loss.

    Returns the full results dict that also gets written to `output_path`.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr_queries, precision_queries, n_overlap_excluded = build_query_sets(manifest)
    print(
        f"Query sets: {len(asr_queries)} ASR (attack) queries, "
        f"{len(precision_queries)} precision (clean) queries "
        f"({n_overlap_excluded} overlapping rows excluded from precision)."
    )

    embedder = Embedder()
    client = get_client()
    tracker = FrequencyTracker()

    print(f"Running defended retrieval (top_k={config.retrieval_top_k})...")
    defended_precision_scores = []
    for i, query in enumerate(precision_queries):
        result = retrieve(
            query.question,
            config=config,
            embedder=embedder,
            qdrant_client_instance=client,
            tracker=tracker,
        )
        chunk_ids = [gated.candidate.chunk_id for gated in result.candidates]
        defended_precision_scores.append(_precision_at_k(chunk_ids, query.relevant_chunk_ids))
        if (i + 1) % 20 == 0 or (i + 1) == len(precision_queries):
            print(f"  defended precision: {i + 1}/{len(precision_queries)}")

    defended_asr_hits = []
    for i, query in enumerate(asr_queries):
        result = retrieve(
            query.question,
            config=config,
            embedder=embedder,
            qdrant_client_instance=client,
            tracker=tracker,
        )
        chunk_ids = [gated.candidate.chunk_id for gated in result.candidates]
        defended_asr_hits.append(_attack_succeeded(chunk_ids, query.poison_chunk_ids))
        if (i + 1) % 20 == 0 or (i + 1) == len(asr_queries):
            print(f"  defended ASR: {i + 1}/{len(asr_queries)}")

    defended_asr = sum(defended_asr_hits) / len(defended_asr_hits) if defended_asr_hits else 0.0
    defended_precision = (
        sum(defended_precision_scores) / len(defended_precision_scores)
        if defended_precision_scores
        else 0.0
    )
    metrics: dict = {"defended": {"asr": defended_asr, "precision_at_k": defended_precision}}
    exit_criteria = None

    if compare_baseline:
        print("Running undefended baseline...")
        undefended_asr_hits = [
            _attack_succeeded(
                _undefended_chunk_ids(query.question, embedder, client, config),
                query.poison_chunk_ids,
            )
            for query in asr_queries
        ]
        undefended_precision_scores = [
            _precision_at_k(
                _undefended_chunk_ids(query.question, embedder, client, config),
                query.relevant_chunk_ids,
            )
            for query in precision_queries
        ]
        undefended_asr = (
            sum(undefended_asr_hits) / len(undefended_asr_hits) if undefended_asr_hits else 0.0
        )
        undefended_precision = (
            sum(undefended_precision_scores) / len(undefended_precision_scores)
            if undefended_precision_scores
            else 0.0
        )

        asr_reduction_pp = 100.0 * (undefended_asr - defended_asr)
        precision_loss_pp = 100.0 * (undefended_precision - defended_precision)
        precision_loss_relative_pct = (
            100.0 * (undefended_precision - defended_precision) / undefended_precision
            if undefended_precision > 0
            else 0.0
        )

        metrics["undefended"] = {"asr": undefended_asr, "precision_at_k": undefended_precision}
        metrics["deltas"] = {
            "asr_reduction_pp": asr_reduction_pp,
            "precision_loss_pp": precision_loss_pp,
            "precision_loss_relative_pct": precision_loss_relative_pct,
        }
        passed = (
            asr_reduction_pp >= config.exit_criteria_asr_reduction_pp_min
            and precision_loss_relative_pct < config.exit_criteria_precision_loss_relative_pct_max
        )
        exit_criteria = {
            "asr_reduction_pp_min": config.exit_criteria_asr_reduction_pp_min,
            "precision_loss_relative_pct_max": config.exit_criteria_precision_loss_relative_pct_max,
            "passed": passed,
        }

    print()
    _print_retrieval_report(metrics)
    if exit_criteria is not None:
        verdict = "PASSED" if exit_criteria["passed"] else "FAILED"
        print(
            f"\nExit criteria: {verdict} "
            f"(ASR reduction {metrics['deltas']['asr_reduction_pp']:.1f}pp "
            f">= {exit_criteria['asr_reduction_pp_min']}pp required; "
            f"precision loss {metrics['deltas']['precision_loss_relative_pct']:.1f}% "
            f"< {exit_criteria['precision_loss_relative_pct_max']}% required)"
        )

    results: dict = {
        "manifest_path": str(manifest_path),
        "n_chunks": len(manifest),
        "n_asr_queries": len(asr_queries),
        "n_precision_queries": len(precision_queries),
        "n_overlap_excluded": n_overlap_excluded,
        "config": asdict(config),
        "metrics": metrics,
    }
    if exit_criteria is not None:
        results["exit_criteria"] = exit_criteria

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output_path}")

    return results


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


def _print_generation_report(metrics: dict) -> None:
    print(f"{'generation_asr':<28}{metrics['generation_asr']:>10.3f}")
    print(f"{'faithfulness_precision':<28}{metrics['faithfulness_precision']:>10.3f}")
    print(
        f"n_asr_queries={metrics['n_asr_queries']}  "
        f"n_precision_queries={metrics['n_precision_queries']}"
    )
    print()
    print(f"{'latency (ms)':<20}{'mean':>10}{'p50':>10}{'p95':>10}")
    for name, stage in metrics["latency_ms"].items():
        print(f"{name:<20}{stage['mean']:>10.1f}{stage['p50']:>10.1f}{stage['p95']:>10.1f}")


def run_generation_eval(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_GENERATION_OUTPUT_PATH,
    policy_config: PolicyConfig = DEFAULT_POLICY_CONFIG,
    generation_config: GenerationConfig = DEFAULT_GEN_CONFIG,
    faithfulness_config: FaithfulnessConfig = DEFAULT_FAITHFULNESS_CONFIG,
    asr_sample_size: int | None = None,
    precision_sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the Phase 5 generation-stage eval: end-to-end (retrieve + generate)
    ASR at the *answer* level, faithfulness precision, and latency.

    `generation_asr` reuses the same manifest ground truth (`poison_chunk_ids`)
    as the Phase 4 retrieval eval, but the bar is higher: an attack only
    counts as succeeding here if a generated claim is NLI-entailed by a
    poisoned chunk (via `FaithfulnessChecker`'s `best_chunk_id`), i.e. the
    poison didn't just survive retrieval gating -- it actually grounded part
    of the final answer. `faithfulness_precision` is the standard RAG
    groundedness metric (fraction of generated claims entailed by *some*
    retrieved chunk), measured over the clean/precision query set. Calls the
    real configured LLM provider once per query -- costs real API budget;
    use `asr_sample_size`/`precision_sample_size` to cap it, same pattern as
    `run_detection_eval`'s `judge_sample_size`.

    Known measurement caveat, observed live in a Phase 5 verification run
    (`faithfulness_precision` = 0.425 on a 20-query sample) rather than
    theoretical: `GenerationConfig.generation_system_prompt` explicitly asks
    the model to say so when context is insufficient, and the retrieval
    layer's own precision@k (~0.35, see Phase 4 numbers in PROGRESS.md) means
    many precision/clean queries genuinely don't surface their one relevant
    chunk in the top-k. When that happens the model correctly abstains (e.g.
    "the provided context does not contain..."), but that abstention
    sentence is itself a "claim" under `split_claims` and NLI correctly finds
    no chunk entailing it -- so a *correct refusal* gets scored identically
    to a *hallucination* here. `faithfulness_precision` as currently measured
    is therefore a lower bound on true groundedness, not a hallucination
    rate; a fix would need an abstention classifier ahead of the NLI check,
    which risks its own false-negatives (real hallucinations that happen to
    contain hedging language) and is left as future work rather than a
    heuristic bolted on under this phase's scope.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr_queries, precision_queries, n_overlap_excluded = build_query_sets(manifest)

    rng = random.Random(seed)
    if asr_sample_size is not None and asr_sample_size < len(asr_queries):
        idx = sorted(rng.sample(range(len(asr_queries)), asr_sample_size))
        asr_queries = [asr_queries[i] for i in idx]
    if precision_sample_size is not None and precision_sample_size < len(precision_queries):
        idx = sorted(rng.sample(range(len(precision_queries)), precision_sample_size))
        precision_queries = [precision_queries[i] for i in idx]

    print(
        f"Generation eval: {len(asr_queries)} ASR queries, "
        f"{len(precision_queries)} precision queries ({n_overlap_excluded} overlap excluded)."
    )

    embedder = Embedder()
    client = get_client()
    tracker = FrequencyTracker()
    llm = get_llm(temperature=generation_config.generation_temperature)
    faithfulness_checker = FaithfulnessChecker(faithfulness_config)

    retrieval_latencies_ms: list[float] = []
    generation_latencies_ms: list[float] = []

    def _run_query(question: str):
        t0 = time.perf_counter()
        result = retrieve(
            question,
            config=policy_config,
            embedder=embedder,
            qdrant_client_instance=client,
            tracker=tracker,
        )
        t1 = time.perf_counter()
        gen = generate(
            question,
            result,
            config=generation_config,
            faithfulness_config=faithfulness_config,
            llm=llm,
            faithfulness_checker=faithfulness_checker,
        )
        t2 = time.perf_counter()
        retrieval_latencies_ms.append((t1 - t0) * 1000)
        generation_latencies_ms.append((t2 - t1) * 1000)
        return result, gen

    print("Running ASR (attack) queries: retrieve + generate...")
    asr_hits = []
    for i, q in enumerate(asr_queries):
        _, gen = _run_query(q.question)
        asr_hits.append(_generation_attack_succeeded(gen, q.poison_chunk_ids))
        if (i + 1) % 5 == 0 or (i + 1) == len(asr_queries):
            print(f"  generation ASR: {i + 1}/{len(asr_queries)}")

    print("Running precision (clean) queries: retrieve + generate...")
    faithfulness_scores = []
    for i, q in enumerate(precision_queries):
        _, gen = _run_query(q.question)
        if gen.faithfulness is not None:
            faithfulness_scores.append(gen.faithfulness.faithfulness_score)
        if (i + 1) % 10 == 0 or (i + 1) == len(precision_queries):
            print(f"  generation precision: {i + 1}/{len(precision_queries)}")

    generation_asr = sum(asr_hits) / len(asr_hits) if asr_hits else 0.0
    faithfulness_precision = (
        sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    )
    total_latencies_ms = [
        r + g for r, g in zip(retrieval_latencies_ms, generation_latencies_ms, strict=True)
    ]

    def _stage_latency(values: list[float]) -> dict[str, float]:
        return {
            "mean": fmean(values) if values else 0.0,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
        }

    metrics = {
        "generation_asr": generation_asr,
        "faithfulness_precision": faithfulness_precision,
        "n_asr_queries": len(asr_queries),
        "n_precision_queries": len(precision_queries),
        "latency_ms": {
            "retrieval": _stage_latency(retrieval_latencies_ms),
            "generation": _stage_latency(generation_latencies_ms),
            "end_to_end": _stage_latency(total_latencies_ms),
        },
    }

    print()
    _print_generation_report(metrics)

    results = {
        "manifest_path": str(manifest_path),
        "n_chunks": len(manifest),
        "n_overlap_excluded": n_overlap_excluded,
        "config": {
            "generation": asdict(generation_config),
            "faithfulness": asdict(faithfulness_config),
        },
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output_path}")

    return results


def _load_baseline_results(path: Path) -> dict | None:
    """Load a previously-persisted eval-results JSON file, or None if it
    doesn't exist yet (e.g. a fresh clone that hasn't run Phase 4/5)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _print_agentic_report(metrics: dict) -> None:
    print(f"{'avg_subquery_count':<34}{metrics['avg_subquery_count']:>10.2f}")
    print(
        f"n_asr_queries={metrics['n_asr_queries']}  "
        f"n_precision_queries={metrics['n_precision_queries']}"
    )
    print()

    retrieval = metrics["retrieval"]
    print(f"{'retrieval asr (this run)':<34}{retrieval['asr']:>10.3f}")
    print(f"{'retrieval precision@k (this run)':<34}{retrieval['precision_at_k']:>10.3f}")
    baseline = retrieval.get("baseline")
    if baseline is None:
        print("  (no Phase 4 baseline found at eval/benchmarks/retrieval_eval_results.json)")
    else:
        if (
            baseline["n_asr_queries"] != metrics["n_asr_queries"]
            or baseline["n_precision_queries"] != metrics["n_precision_queries"]
        ):
            print(
                f"  WARNING: baseline sample sizes differ (n_asr={baseline['n_asr_queries']}, "
                f"n_precision={baseline['n_precision_queries']}) -- deltas below are not "
                "strictly apples-to-apples"
            )
        print(
            f"{'  baseline asr':<34}{baseline['asr']:>10.3f}   delta {retrieval['asr_delta']:+.3f}"
        )
        print(
            f"{'  baseline precision@k':<34}{baseline['precision_at_k']:>10.3f}"
            f"   delta {retrieval['precision_at_k_delta']:+.3f}"
        )

    print()
    generation = metrics.get("generation")
    if generation is None:
        print("generation stage skipped (--skip-generation)")
    else:
        print(f"{'generation asr (this run)':<34}{generation['generation_asr']:>10.3f}")
        print(
            f"{'faithfulness precision (this run)':<34}"
            f"{generation['faithfulness_precision']:>10.3f}"
        )
        gen_baseline = generation.get("baseline")
        if gen_baseline is None:
            print("  (no Phase 5 baseline found at eval/benchmarks/generation_eval_results.json)")
        else:
            print(
                f"{'  baseline generation asr':<34}{gen_baseline['generation_asr']:>10.3f}"
                f"   delta {generation['generation_asr_delta']:+.3f}"
            )
            print(
                f"{'  baseline faithfulness precision':<34}"
                f"{gen_baseline['faithfulness_precision']:>10.3f}"
                f"   delta {generation['faithfulness_precision_delta']:+.3f}"
            )

    print()
    print(metrics["detection_f1_note"])


def run_agentic_eval(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_AGENTIC_OUTPUT_PATH,
    policy_config: PolicyConfig = DEFAULT_POLICY_CONFIG,
    generation_config: GenerationConfig = DEFAULT_GEN_CONFIG,
    faithfulness_config: FaithfulnessConfig = DEFAULT_FAITHFULNESS_CONFIG,
    agentic_config: AgenticConfig = DEFAULT_AGENTIC_CONFIG,
    asr_sample_size: int | None = None,
    precision_sample_size: int | None = None,
    skip_generation: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Phase 8 agentic-RAG eval: re-runs the Phase 4/5 poisoning eval through
    the planner -> parallel retriever agents -> synthesizer pipeline
    (`src/agentic/{planner,graph}.py`) instead of a single `retrieve()` +
    `generate()` call, and reports ASR/precision deltas against the
    persisted Phase 4/5 baselines.

    One combined pass per query, unlike Phase 4/5's split retrieval/
    generation stages -- every agentic query costs a planner LLM call
    regardless of which metrics you want, so splitting the stages buys
    nothing here.

    Detection F1 is NOT re-run or re-reported: `DetectionPipeline` scoring
    runs identically inside the unchanged `retrieve()` call on every
    fan-out branch, so Phase 3's persisted `detection_eval_results.json` F1
    still applies verbatim -- see `detection_f1_note` in the returned dict.

    Expected, correct outcome on this repo's single-hop NQ corpus (not a
    failed experiment): the planner should mostly degrade to a single
    sub-query (`avg_subquery_count` close to 1.0), so ASR/precision deltas
    vs. the Phase 4/5 baselines should sit near zero -- confirming the
    agentic wrapper doesn't erode Phase 4's defense guarantee, rather than
    "the agentic layer changed nothing" being a bug.

    `skip_generation=True` runs the planner + parallel-retrieval + merge legs
    only, with zero LLM calls anywhere -- the live-verifiable path in an
    environment where `OPENAI_API_KEY` has no billing quota, mirroring
    `--skip-judge`'s precedent for running a stage without a component that
    needs unavailable resources.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr_queries, precision_queries, n_overlap_excluded = build_query_sets(manifest)

    rng = random.Random(seed)
    if asr_sample_size is not None and asr_sample_size < len(asr_queries):
        idx = sorted(rng.sample(range(len(asr_queries)), asr_sample_size))
        asr_queries = [asr_queries[i] for i in idx]
    if precision_sample_size is not None and precision_sample_size < len(precision_queries):
        idx = sorted(rng.sample(range(len(precision_queries)), precision_sample_size))
        precision_queries = [precision_queries[i] for i in idx]

    print(
        f"Agentic eval: {len(asr_queries)} ASR queries, "
        f"{len(precision_queries)} precision queries ({n_overlap_excluded} overlap excluded)."
    )

    embedder = Embedder()
    client = get_client()
    tracker = FrequencyTracker()
    llm = None if skip_generation else get_llm(temperature=generation_config.generation_temperature)
    faithfulness_checker = None if skip_generation else FaithfulnessChecker(faithfulness_config)

    subquery_counts: list[int] = []

    def _run_agentic_query(question: str) -> tuple[list[str], GenerationResult | None]:
        sub_queries = plan_subqueries(question, config=agentic_config)
        subquery_counts.append(len(sub_queries))
        sub_results = [
            retrieve(
                sub_query,
                config=policy_config,
                embedder=embedder,
                qdrant_client_instance=client,
                tracker=tracker,
            )
            for sub_query in sub_queries
        ]
        merged = merge_sub_results(question, sub_results, config=policy_config)
        chunk_ids = [gated.candidate.chunk_id for gated in merged.candidates]
        generation = None
        if not skip_generation:
            generation = generate(
                question,
                merged,
                config=generation_config,
                faithfulness_config=faithfulness_config,
                llm=llm,
                faithfulness_checker=faithfulness_checker,
            )
        return chunk_ids, generation

    action = "plan + parallel retrieve + merge" + ("" if skip_generation else " + generate")
    print(f"Running ASR (attack) queries: {action}...")
    asr_hits: list[bool] = []
    generation_asr_hits: list[bool] = []
    for i, q in enumerate(asr_queries):
        chunk_ids, generation = _run_agentic_query(q.question)
        asr_hits.append(_attack_succeeded(chunk_ids, q.poison_chunk_ids))
        if generation is not None:
            generation_asr_hits.append(_generation_attack_succeeded(generation, q.poison_chunk_ids))
        if (i + 1) % 5 == 0 or (i + 1) == len(asr_queries):
            print(f"  agentic ASR: {i + 1}/{len(asr_queries)}")

    print(f"Running precision (clean) queries: {action}...")
    precision_scores: list[float] = []
    faithfulness_scores: list[float] = []
    for i, q in enumerate(precision_queries):
        chunk_ids, generation = _run_agentic_query(q.question)
        precision_scores.append(_precision_at_k(chunk_ids, q.relevant_chunk_ids))
        if generation is not None and generation.faithfulness is not None:
            faithfulness_scores.append(generation.faithfulness.faithfulness_score)
        if (i + 1) % 10 == 0 or (i + 1) == len(precision_queries):
            print(f"  agentic precision: {i + 1}/{len(precision_queries)}")

    asr = sum(asr_hits) / len(asr_hits) if asr_hits else 0.0
    precision_at_k = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_subquery_count = fmean(subquery_counts) if subquery_counts else 0.0

    retrieval_metrics: dict = {"asr": asr, "precision_at_k": precision_at_k}
    retrieval_baseline = _load_baseline_results(DEFAULT_RETRIEVAL_OUTPUT_PATH)
    if retrieval_baseline is not None and "defended" in retrieval_baseline.get("metrics", {}):
        defended = retrieval_baseline["metrics"]["defended"]
        retrieval_metrics["baseline"] = {
            "asr": defended["asr"],
            "precision_at_k": defended["precision_at_k"],
            "n_asr_queries": retrieval_baseline["n_asr_queries"],
            "n_precision_queries": retrieval_baseline["n_precision_queries"],
        }
        retrieval_metrics["asr_delta"] = asr - defended["asr"]
        retrieval_metrics["precision_at_k_delta"] = precision_at_k - defended["precision_at_k"]

    generation_metrics: dict | None = None
    if not skip_generation:
        generation_asr = (
            sum(generation_asr_hits) / len(generation_asr_hits) if generation_asr_hits else 0.0
        )
        faithfulness_precision = (
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        )
        generation_metrics = {
            "generation_asr": generation_asr,
            "faithfulness_precision": faithfulness_precision,
        }
        generation_baseline = _load_baseline_results(DEFAULT_GENERATION_OUTPUT_PATH)
        if generation_baseline is not None:
            baseline_gen_metrics = generation_baseline["metrics"]
            generation_metrics["baseline"] = {
                "generation_asr": baseline_gen_metrics["generation_asr"],
                "faithfulness_precision": baseline_gen_metrics["faithfulness_precision"],
                "n_asr_queries": baseline_gen_metrics["n_asr_queries"],
                "n_precision_queries": baseline_gen_metrics["n_precision_queries"],
            }
            generation_metrics["generation_asr_delta"] = (
                generation_asr - baseline_gen_metrics["generation_asr"]
            )
            generation_metrics["faithfulness_precision_delta"] = (
                faithfulness_precision - baseline_gen_metrics["faithfulness_precision"]
            )

    metrics = {
        "avg_subquery_count": avg_subquery_count,
        "n_asr_queries": len(asr_queries),
        "n_precision_queries": len(precision_queries),
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "detection_f1_note": (
            "Not re-run: DetectionPipeline scoring runs identically inside the unchanged "
            "retrieve() call on every fan-out branch, so Phase 3's persisted "
            "eval/benchmarks/detection_eval_results.json F1 still applies verbatim."
        ),
    }

    print()
    _print_agentic_report(metrics)

    results = {
        "manifest_path": str(manifest_path),
        "n_chunks": len(manifest),
        "n_overlap_excluded": n_overlap_excluded,
        "skip_generation": skip_generation,
        "config": {
            "agentic": asdict(agentic_config),
            "policy": asdict(policy_config),
        },
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output_path}")

    return results


def _print_unlearning_report(metrics: dict) -> None:
    print(f"{'latency (ms)':<24}{'mean':>10}{'p50':>10}{'p95':>10}")
    for name, stage in metrics["latency_ms"].items():
        print(f"{name:<24}{stage['mean']:>10.1f}{stage['p50']:>10.1f}{stage['p95']:>10.1f}")
    print()
    print(f"{'receipt_validity_rate':<36}{metrics['receipt_validity_rate']:>10.3f}")
    print(f"{'membership_check_pass_rate':<36}{metrics['membership_check_pass_rate']:>10.3f}")
    print(
        f"{'reconstruction_attack_success_rate':<36}"
        f"{metrics['reconstruction_attack_success_rate']:>10.3f}"
    )
    rp = metrics["removal_precision_at_k"]
    print(f"{'removal_precision@k before':<36}{rp['before']:>10.3f}")
    print(f"{'removal_precision@k after':<36}{rp['after']:>10.3f}")


def run_unlearning_eval(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_UNLEARNING_OUTPUT_PATH,
    config: UnlearningConfig = DEFAULT_UNLEARNING_CONFIG,
    policy_config: PolicyConfig = DEFAULT_POLICY_CONFIG,
    sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
    requester: str = "eval-harness",
) -> dict:
    """Phase 6 unlearning eval: deletes a seeded random sample of chunks
    against a disposable scratch collection -- `UNLEARNING_EVAL_SCRATCH_COLLECTION`,
    always torn down in a `finally`, never the live corpus -- and benchmarks
    deletion latency, receipt gen/verify time and validity, the ghost-vector-
    defense membership-inference check, the corpus-level reconstruction-
    attack success rate vs. chance baseline, and removal-precision@k
    before/after each deletion: the full set of Table VI metrics `results.md`
    is waiting on.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr_queries, precision_queries, _ = build_query_sets(manifest)
    chunk_to_question = _chunk_to_question_map(asr_queries, precision_queries)

    rng = random.Random(seed)
    n_sample = min(sample_size or config.eval_sample_size, len(chunk_to_question))
    sampled_chunk_ids = rng.sample(sorted(chunk_to_question), n_sample)

    embedder = Embedder()
    client = get_client()
    signer = ProvenanceSigner()

    if client.collection_exists(UNLEARNING_EVAL_SCRATCH_COLLECTION):
        client.delete_collection(UNLEARNING_EVAL_SCRATCH_COLLECTION)
    upsert_manifest(
        client, manifest, embedder, signer, collection_name=UNLEARNING_EVAL_SCRATCH_COLLECTION
    )

    delete_latencies: list[float] = []
    compaction_latencies: list[float] = []
    total_latencies: list[float] = []
    receipt_gen_latencies: list[float] = []
    receipt_verify_latencies: list[float] = []
    receipt_valid_flags: list[bool] = []
    membership_pass_flags: list[bool] = []
    reconstruction_exceeds_flags: list[bool] = []
    precision_before: list[float] = []
    precision_after: list[float] = []

    try:
        for i, chunk_id in enumerate(sampled_chunk_ids):
            question = chunk_to_question[chunk_id]
            query_vector = embedder.embed_one(question).tolist()

            hits_before = search(
                client,
                query_vector,
                collection_name=UNLEARNING_EVAL_SCRATCH_COLLECTION,
                top_k=policy_config.retrieval_top_k,
            )
            chunk_ids_before = [hit.payload["chunk_id"] for hit in hits_before]
            precision_before.append(_precision_at_k(chunk_ids_before, frozenset({chunk_id})))

            bundle = delete_with_receipt(
                client,
                chunk_id,
                requester=requester,
                collection_name=UNLEARNING_EVAL_SCRATCH_COLLECTION,
                config=config,
                signer=signer,
                run_reconstruction_check=True,
                decoy_seed=seed,
            )

            hits_after = search(
                client,
                query_vector,
                collection_name=UNLEARNING_EVAL_SCRATCH_COLLECTION,
                top_k=policy_config.retrieval_top_k,
            )
            chunk_ids_after = [hit.payload["chunk_id"] for hit in hits_after]
            precision_after.append(_precision_at_k(chunk_ids_after, frozenset({chunk_id})))

            deletion = bundle.deletion
            delete_latencies.append(deletion.delete_latency_ms)
            if deletion.compaction_latency_ms is not None:
                compaction_latencies.append(deletion.compaction_latency_ms)
                total_latencies.append(deletion.delete_latency_ms + deletion.compaction_latency_ms)
            else:
                total_latencies.append(deletion.delete_latency_ms)
            receipt_gen_latencies.append(bundle.receipt_gen_latency_ms)
            receipt_verify_latencies.append(bundle.receipt_verify_latency_ms)
            receipt_valid_flags.append(bundle.receipt_valid)
            if bundle.membership_check_passed is not None:
                membership_pass_flags.append(bundle.membership_check_passed)
            if bundle.reconstruction_check is not None:
                reconstruction_exceeds_flags.append(bundle.reconstruction_check.exceeds_chance)

            if (i + 1) % 5 == 0 or (i + 1) == len(sampled_chunk_ids):
                print(f"  deleted: {i + 1}/{len(sampled_chunk_ids)}")
    finally:
        client.delete_collection(UNLEARNING_EVAL_SCRATCH_COLLECTION)

    def _stage_latency(values: list[float]) -> dict[str, float]:
        return {
            "mean": fmean(values) if values else 0.0,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
        }

    metrics = {
        "n_deleted": len(sampled_chunk_ids),
        "latency_ms": {
            "delete_latency_ms": _stage_latency(delete_latencies),
            "compaction_latency_ms": _stage_latency(compaction_latencies),
            "total_deletion_latency_ms": _stage_latency(total_latencies),
            "receipt_gen_latency_ms": _stage_latency(receipt_gen_latencies),
            "receipt_verify_latency_ms": _stage_latency(receipt_verify_latencies),
        },
        "receipt_validity_rate": (
            sum(receipt_valid_flags) / len(receipt_valid_flags) if receipt_valid_flags else 0.0
        ),
        "membership_check_pass_rate": (
            sum(membership_pass_flags) / len(membership_pass_flags)
            if membership_pass_flags
            else 0.0
        ),
        "reconstruction_attack_success_rate": (
            sum(reconstruction_exceeds_flags) / len(reconstruction_exceeds_flags)
            if reconstruction_exceeds_flags
            else 0.0
        ),
        "removal_precision_at_k": {
            "before": fmean(precision_before) if precision_before else 0.0,
            "after": fmean(precision_after) if precision_after else 0.0,
            "top_k": policy_config.retrieval_top_k,
        },
    }

    print()
    _print_unlearning_report(metrics)

    results = {
        "manifest_path": str(manifest_path),
        "n_chunks": len(manifest),
        "n_eligible_for_sampling": len(chunk_to_question),
        "seed": seed,
        "config": asdict(config),
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output_path}")

    return results


def run_full_report(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_FULL_REPORT_OUTPUT_PATH,
    include_judge: bool = False,
    judge_sample_size: int | None = None,
    generation_asr_sample_size: int | None = None,
    generation_precision_sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run detection + retrieval + generation end to end and print one
    combined report covering every metric PLAN.md Phase 5 asks for: F1 comes
    from the detection stage, ASR/precision@k from the retrieval stage, and
    end-to-end ASR/faithfulness-precision/latency from the generation stage.
    No single invented cross-stage metric -- each number is already the
    natural metric for its own stage; `--full-report` just runs all three and
    reports them together. The LLM-judge baseline is skipped by default
    (`include_judge=False`) since it's a comparison baseline, not part of the
    defended pipeline, and costs its own separate API budget on top of the
    generation stage's real LLM calls.
    """
    print("=" * 72)
    print("STAGE 1/3 -- Detection")
    print("=" * 72)
    detection_results = run_detection_eval(
        manifest_path=manifest_path,
        output_path=DEFAULT_OUTPUT_PATH,
        classifier_path=DEFAULT_MODEL_PATH,
        skip_judge=not include_judge,
        judge_sample_size=judge_sample_size,
        seed=seed,
    )

    print()
    print("=" * 72)
    print("STAGE 2/3 -- Retrieval")
    print("=" * 72)
    retrieval_results = run_retrieval_eval(
        manifest_path=manifest_path,
        output_path=DEFAULT_RETRIEVAL_OUTPUT_PATH,
        compare_baseline=True,
    )

    print()
    print("=" * 72)
    print("STAGE 3/3 -- Generation")
    print("=" * 72)
    generation_results = run_generation_eval(
        manifest_path=manifest_path,
        output_path=DEFAULT_GENERATION_OUTPUT_PATH,
        asr_sample_size=generation_asr_sample_size,
        precision_sample_size=generation_precision_sample_size,
        seed=seed,
    )

    full_results = {
        "manifest_path": str(manifest_path),
        "detection": detection_results,
        "retrieval": retrieval_results,
        "generation": generation_results,
    }

    detection_metrics = detection_results["metrics"]
    retrieval_metrics = retrieval_results["metrics"]
    generation_metrics = generation_results["metrics"]

    print()
    print("=" * 72)
    print("FULL REPORT SUMMARY")
    print("=" * 72)
    print(f"Detection fused-pipeline F1:       {detection_metrics['fused_pipeline']['f1']:.3f}")
    if "llm_judge" in detection_metrics:
        print(f"Detection LLM-judge F1:            {detection_metrics['llm_judge']['f1']:.3f}")
    print(f"Retrieval ASR (defended):          {retrieval_metrics['defended']['asr']:.3f}")
    precision_at_k = retrieval_metrics["defended"]["precision_at_k"]
    print(f"Retrieval precision@k (defended):  {precision_at_k:.3f}")
    if "deltas" in retrieval_metrics:
        asr_reduction = retrieval_metrics["deltas"]["asr_reduction_pp"]
        print(f"Retrieval ASR reduction (pp):      {asr_reduction:+.1f}")
    print(f"Generation ASR (end-to-end):       {generation_metrics['generation_asr']:.3f}")
    print(f"Generation faithfulness precision: {generation_metrics['faithfulness_precision']:.3f}")
    end_to_end_mean_ms = generation_metrics["latency_ms"]["end_to_end"]["mean"]
    print(f"End-to-end latency mean (ms):      {end_to_end_mean_ms:.1f}")
    if "exit_criteria" in retrieval_results:
        verdict = "PASSED" if retrieval_results["exit_criteria"]["passed"] else "FAILED"
        print(f"Phase 4 exit criteria (ASR reduction / precision loss): {verdict}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"\nWrote full report to {output_path}")

    return full_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["detection", "retrieval", "generation", "unlearning", "agentic"],
        default="detection",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Run detection + retrieval + generation end to end and print/write one "
        "combined ASR / precision / F1 / latency report (ignores --stage).",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--classifier-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--skip-judge", action="store_true", help="[detection] skip the LLM-judge baseline."
    )
    parser.add_argument(
        "--include-judge",
        action="store_true",
        help="[full-report] also run the LLM-judge baseline (skipped by default -- "
        "separate API budget on top of the generation stage's own LLM calls).",
    )
    parser.add_argument(
        "--judge-sample-size",
        type=int,
        default=None,
        help="[detection/full-report] judge only a seeded random subsample instead of the "
        "full set.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="[retrieval] also run the undefended baseline and report ASR-reduction / "
        "precision-loss deltas plus the exit-criteria verdict.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="[retrieval] override PolicyConfig.retrieval_top_k for this run.",
    )
    parser.add_argument(
        "--generation-asr-sample-size",
        type=int,
        default=None,
        help="[generation/full-report] evaluate only a seeded random subsample of ASR "
        "queries instead of the full set (each query is a real LLM call).",
    )
    parser.add_argument(
        "--generation-precision-sample-size",
        type=int,
        default=None,
        help="[generation/full-report] evaluate only a seeded random subsample of "
        "precision queries instead of the full set (each query is a real LLM call).",
    )
    parser.add_argument(
        "--unlearning-sample-size",
        type=int,
        default=None,
        help="[unlearning] delete only a seeded random subsample instead of "
        "UnlearningConfig.eval_sample_size.",
    )
    parser.add_argument(
        "--agentic-asr-sample-size",
        type=int,
        default=5,
        help="[agentic] evaluate only a seeded random subsample of ASR queries instead of "
        "the full set. Defaults to a small number (unlike every other stage's None=full-set "
        "default) since a fan-out of up to AgenticConfig.agentic_max_subqueries sub-queries "
        "multiplies Phase 5's already ~10s/leg retrieval latency; pass 0 or a value >= the "
        "full query count to run everything.",
    )
    parser.add_argument(
        "--agentic-precision-sample-size",
        type=int,
        default=10,
        help="[agentic] evaluate only a seeded random subsample of precision queries "
        "instead of the full set. Same small-default reasoning as "
        "--agentic-asr-sample-size.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="[agentic] run the planner + parallel-retrieval + merge legs only, with zero "
        "LLM calls anywhere (mirrors --skip-judge's precedent). The live-verifiable path "
        "in an environment where OPENAI_API_KEY has no billing quota -- without this flag, "
        "--stage agentic will fail with the same RateLimitError already documented for "
        "Phase 5/7's generation calls.",
    )
    args = parser.parse_args()

    if args.full_report:
        run_full_report(
            manifest_path=args.manifest,
            output_path=args.output or DEFAULT_FULL_REPORT_OUTPUT_PATH,
            include_judge=args.include_judge,
            judge_sample_size=args.judge_sample_size,
            generation_asr_sample_size=args.generation_asr_sample_size,
            generation_precision_sample_size=args.generation_precision_sample_size,
            seed=args.seed,
        )
    elif args.stage == "detection":
        run_detection_eval(
            manifest_path=args.manifest,
            output_path=args.output or DEFAULT_OUTPUT_PATH,
            classifier_path=args.classifier_path,
            skip_judge=args.skip_judge,
            judge_sample_size=args.judge_sample_size,
            seed=args.seed,
        )
    elif args.stage == "retrieval":
        config = (
            DEFAULT_POLICY_CONFIG
            if args.top_k is None
            else PolicyConfig(retrieval_top_k=args.top_k)
        )
        run_retrieval_eval(
            manifest_path=args.manifest,
            output_path=args.output or DEFAULT_RETRIEVAL_OUTPUT_PATH,
            compare_baseline=args.compare_baseline,
            config=config,
        )
    elif args.stage == "unlearning":
        run_unlearning_eval(
            manifest_path=args.manifest,
            output_path=args.output or DEFAULT_UNLEARNING_OUTPUT_PATH,
            sample_size=args.unlearning_sample_size,
            seed=args.seed,
        )
    elif args.stage == "agentic":
        run_agentic_eval(
            manifest_path=args.manifest,
            output_path=args.output or DEFAULT_AGENTIC_OUTPUT_PATH,
            asr_sample_size=args.agentic_asr_sample_size or None,
            precision_sample_size=args.agentic_precision_sample_size or None,
            skip_generation=args.skip_generation,
            seed=args.seed,
        )
    else:
        run_generation_eval(
            manifest_path=args.manifest,
            output_path=args.output or DEFAULT_GENERATION_OUTPUT_PATH,
            asr_sample_size=args.generation_asr_sample_size,
            precision_sample_size=args.generation_precision_sample_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
