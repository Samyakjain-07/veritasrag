"""Phase 9 cost-accuracy frontier: toggles each Phase 4/5 defensive layer
off individually (leave-one-out ablation) against the shared Phase 4 ASR /
precision@k query sets, measuring each configuration's real cost (retrieval
wall-clock latency) alongside its accuracy (ASR, precision@k) -- the
frontier PLAN.md's Phase 9 prompt asks for, and the concrete use case
CLAUDE.md's "don't hardcode thresholds ... so they can be swept for the
cost-accuracy frontier report in Phase 9" note was written for.

Layers toggled (see `src/retrieval/config.py`'s `PolicyConfig.enable_*`
fields, added this phase specifically so disabling a layer also skips its
*compute*, not just its triggering threshold -- raising a poison-score
threshold above 1.0 would disable triggering but `retrieve()` would still
pay for the scoring):

  - detection_scoring   -- PolicyConfig.enable_detection_scoring: skips the
                            Phase 3 DetectionPipeline poison-score computation
                            entirely. The single largest cost driver measured
                            in this repo so far (~10s/query, sequential
                            per-candidate perplexity scoring -- see
                            PROGRESS.md's Phase 5/8 latency notes).
  - tier_gating         -- PolicyConfig.enable_tier_gating: ignores
                            trust_tier for both the QUARANTINED categorical
                            drop and the per-tier poison-score band
                            selection (falls back to the unverified band
                            uniformly). Near-zero cost on its own.
  - frequency_tracking  -- PolicyConfig.enable_frequency_tracking: skips the
                            FrequencyTracker record/flag calls. Near-zero
                            cost on its own.
  - faithfulness_check  -- generate()'s existing `run_faithfulness_check`
                            param (Phase 5, no source change needed here):
                            the NLI cross-encoder check at the generation
                            stage. This is a generation-time layer, so it has
                            zero retrieval-level cost/accuracy effect by
                            construction -- its real tradeoff (NLI latency vs.
                            faithfulness_precision / the ability to detect a
                            poisoned claim that grounded an answer) is only
                            measured when `--include-generation` is passed,
                            since it requires a real LLM call per query. The
                            other three layers are 100% local -- no API key
                            or budget needed for the default run.

Plus two reference anchors: `undefended` (raw Qdrant search, no gating at
all -- reuses `eval.run_eval._undefended_chunk_ids`, the same zero-defense
baseline Phase 4's own eval already reports) and `full_defense` (every layer
enabled, the shipped default `PolicyConfig()`).

Each row's `pareto_optimal` flag marks whether any other row is at least as
cheap (latency) AND at least as accurate (ASR no higher, precision@k no
lower) with at least one strict improvement -- a row that's dominated on
every axis by another isn't part of the actual cost-accuracy frontier.

Usage:
    python eval/cost_accuracy_sweep.py
    python eval/cost_accuracy_sweep.py --asr-sample-size 20 --precision-sample-size 106  # full set
    python eval/cost_accuracy_sweep.py --include-generation --generation-asr-sample-size 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean

# Allow running this file directly (`python eval/cost_accuracy_sweep.py`)
# without requiring the repo to be installed as a package -- same
# `pythonpath`-equivalent trick `eval/run_eval.py` already uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SEED,
    RetrievalQuery,
    _attack_succeeded,
    _generation_attack_succeeded,
    _percentile,
    _precision_at_k,
    _undefended_chunk_ids,
    build_query_sets,
)
from src.embeddings.embedder import Embedder  # noqa: E402
from src.generation.config import (  # noqa: E402
    DEFAULT_FAITHFULNESS_CONFIG,
    DEFAULT_GENERATION_CONFIG,
    FaithfulnessConfig,
    GenerationConfig,
)
from src.generation.faithfulness import FaithfulnessChecker  # noqa: E402
from src.generation.generate import generate  # noqa: E402
from src.generation.llm_client import get_llm  # noqa: E402
from src.retrieval.config import DEFAULT_CONFIG as DEFAULT_POLICY_CONFIG  # noqa: E402
from src.retrieval.config import PolicyConfig  # noqa: E402
from src.retrieval.policy_engine import FrequencyTracker, retrieve  # noqa: E402
from src.vectorstore.qdrant_client import get_client  # noqa: E402

DEFAULT_OUTPUT_PATH = Path("eval/benchmarks/cost_accuracy_frontier_results.json")

UNDEFENDED_LABEL = "undefended"
FULL_DEFENSE_LABEL = "full_defense"


@dataclass(frozen=True)
class LayerConfig:
    """One point in the sweep: a name, human-readable description, and the
    `PolicyConfig` to run `retrieve()` with. `run_faithfulness_check` only
    matters for the generation-inclusive part of the sweep."""

    name: str
    description: str
    policy_config: PolicyConfig
    run_faithfulness_check: bool = True


def build_layer_configs(base_policy: PolicyConfig = DEFAULT_POLICY_CONFIG) -> list[LayerConfig]:
    """The leave-one-out ablation grid: full defense, then each layer
    disabled one at a time. Order matters only for report readability
    (cheapest-to-costliest is not guaranteed until latency is measured)."""
    return [
        LayerConfig(
            FULL_DEFENSE_LABEL,
            "Every Phase 4/5 defensive layer enabled (the shipped default).",
            base_policy,
            run_faithfulness_check=True,
        ),
        LayerConfig(
            "no_detection_scoring",
            "Phase 3 poison-score detection/gating disabled.",
            replace(base_policy, enable_detection_scoring=False),
            run_faithfulness_check=True,
        ),
        LayerConfig(
            "no_tier_gating",
            "Trust-tier gating disabled (QUARANTINED drop + per-tier bands ignored).",
            replace(base_policy, enable_tier_gating=False),
            run_faithfulness_check=True,
        ),
        LayerConfig(
            "no_frequency_tracking",
            "Retrieval-frequency anomaly tracking disabled.",
            replace(base_policy, enable_frequency_tracking=False),
            run_faithfulness_check=True,
        ),
        LayerConfig(
            "no_faithfulness_check",
            "Generation-time NLI faithfulness check disabled. Retrieval layer "
            "is identical to full_defense -- this toggle has no retrieval-level "
            "effect by construction; see its generation sub-metrics instead.",
            base_policy,
            run_faithfulness_check=False,
        ),
    ]


def _sample_queries(
    queries: list[RetrievalQuery], sample_size: int | None, seed: int
) -> list[RetrievalQuery]:
    if sample_size is None or sample_size >= len(queries):
        return queries
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(queries)), sample_size))
    return [queries[i] for i in idx]


def _latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values) if values else 0.0,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
    }


def _run_undefended_baseline(
    asr_queries: list[RetrievalQuery],
    precision_queries: list[RetrievalQuery],
    embedder: Embedder,
    client,
    config: PolicyConfig,
) -> dict:
    """Zero-defense reference point: raw Qdrant top-k similarity search, no
    tier filter, no gating, no scoring -- reuses `eval.run_eval`'s own
    undefended-baseline helper so this number matches Phase 4's established
    ASR/precision figures exactly, not a reimplementation that could drift."""
    latencies_ms: list[float] = []
    asr_hits: list[bool] = []
    for query in asr_queries:
        t0 = time.perf_counter()
        chunk_ids = _undefended_chunk_ids(query.question, embedder, client, config)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        asr_hits.append(_attack_succeeded(chunk_ids, query.poison_chunk_ids))

    precision_scores: list[float] = []
    for query in precision_queries:
        t0 = time.perf_counter()
        chunk_ids = _undefended_chunk_ids(query.question, embedder, client, config)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        precision_scores.append(_precision_at_k(chunk_ids, query.relevant_chunk_ids))

    return {
        "asr": fmean(asr_hits) if asr_hits else 0.0,
        "precision_at_k": fmean(precision_scores) if precision_scores else 0.0,
        "latency_ms": _latency_stats(latencies_ms),
    }


def _run_retrieval_layer(
    layer: LayerConfig,
    asr_queries: list[RetrievalQuery],
    precision_queries: list[RetrievalQuery],
    embedder: Embedder,
    client,
) -> dict:
    """Runs `retrieve()` under one `LayerConfig`'s `PolicyConfig`, measuring
    real wall-clock latency per call alongside ASR/precision@k. A fresh
    `FrequencyTracker` per layer -- configs must not leak retrieval-count
    state into each other."""
    tracker = FrequencyTracker()
    latencies_ms: list[float] = []
    asr_hits: list[bool] = []
    for query in asr_queries:
        t0 = time.perf_counter()
        result = retrieve(
            query.question,
            config=layer.policy_config,
            embedder=embedder,
            qdrant_client_instance=client,
            tracker=tracker,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        chunk_ids = [gated.candidate.chunk_id for gated in result.candidates]
        asr_hits.append(_attack_succeeded(chunk_ids, query.poison_chunk_ids))

    precision_scores: list[float] = []
    for query in precision_queries:
        t0 = time.perf_counter()
        result = retrieve(
            query.question,
            config=layer.policy_config,
            embedder=embedder,
            qdrant_client_instance=client,
            tracker=tracker,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        chunk_ids = [gated.candidate.chunk_id for gated in result.candidates]
        precision_scores.append(_precision_at_k(chunk_ids, query.relevant_chunk_ids))

    return {
        "asr": fmean(asr_hits) if asr_hits else 0.0,
        "precision_at_k": fmean(precision_scores) if precision_scores else 0.0,
        "latency_ms": _latency_stats(latencies_ms),
    }


def _run_generation_layer(
    layer: LayerConfig,
    asr_queries: list[RetrievalQuery],
    precision_queries: list[RetrievalQuery],
    embedder: Embedder,
    client,
    generation_config: GenerationConfig,
    faithfulness_config: FaithfulnessConfig,
    llm,
    faithfulness_checker: FaithfulnessChecker | None,
) -> dict:
    """Retrieve + generate for one layer, measuring generation-leg latency
    and the generation-level metrics that only exist post-generation:
    `generation_asr` (a poisoned chunk actually grounded a claim) and
    `faithfulness_precision`. When `layer.run_faithfulness_check=False`,
    `generate()` never runs the NLI check, so `generation.faithfulness` is
    always None -- `generation_asr`/`faithfulness_precision` are reported as
    `None` ("not measurable"), not a misleading 0.0, since disabling this
    layer removes the ability to *detect* grounding, not the grounding
    itself.
    """
    tracker = FrequencyTracker()
    latencies_ms: list[float] = []
    asr_hits: list[bool] = []
    for query in asr_queries:
        result = retrieve(
            query.question,
            config=layer.policy_config,
            embedder=embedder,
            qdrant_client_instance=client,
            tracker=tracker,
        )
        t0 = time.perf_counter()
        gen = generate(
            query.question,
            result,
            config=generation_config,
            faithfulness_config=faithfulness_config,
            llm=llm,
            faithfulness_checker=faithfulness_checker,
            run_faithfulness_check=layer.run_faithfulness_check,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        if gen.faithfulness is not None:
            asr_hits.append(_generation_attack_succeeded(gen, query.poison_chunk_ids))

    faithfulness_scores: list[float] = []
    for query in precision_queries:
        result = retrieve(
            query.question,
            config=layer.policy_config,
            embedder=embedder,
            qdrant_client_instance=client,
            tracker=tracker,
        )
        t0 = time.perf_counter()
        gen = generate(
            query.question,
            result,
            config=generation_config,
            faithfulness_config=faithfulness_config,
            llm=llm,
            faithfulness_checker=faithfulness_checker,
            run_faithfulness_check=layer.run_faithfulness_check,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        if gen.faithfulness is not None:
            faithfulness_scores.append(gen.faithfulness.faithfulness_score)

    generation_asr = fmean(asr_hits) if asr_hits else 0.0
    return {
        "generation_asr": generation_asr if layer.run_faithfulness_check else None,
        "faithfulness_precision": (
            (fmean(faithfulness_scores) if faithfulness_scores else 0.0)
            if layer.run_faithfulness_check
            else None
        ),
        "generation_latency_ms": _latency_stats(latencies_ms),
    }


def _is_dominated(row: dict, all_rows: list[dict]) -> bool:
    """True iff some other row is at least as cheap and at least as
    accurate on both ASR and precision@k, with at least one strict
    improvement -- i.e. `row` contributes nothing to the actual frontier."""
    for other in all_rows:
        if other is row:
            continue
        other_latency, row_latency = other["latency_ms"]["mean"], row["latency_ms"]["mean"]
        at_least_as_good = (
            other_latency <= row_latency
            and other["asr"] <= row["asr"]
            and other["precision_at_k"] >= row["precision_at_k"]
        )
        strictly_better = (
            other_latency < row_latency
            or other["asr"] < row["asr"]
            or other["precision_at_k"] > row["precision_at_k"]
        )
        if at_least_as_good and strictly_better:
            return True
    return False


def _print_report(layers: dict[str, dict], include_generation: bool) -> None:
    header = f"{'layer':<24}{'latency(ms)':>14}{'asr':>10}{'precision@k':>14}{'pareto':>9}"
    print(header)
    print("-" * len(header))
    for name, row in layers.items():
        pareto = "yes" if row.get("pareto_optimal") else ""
        print(
            f"{name:<24}{row['latency_ms']['mean']:>14.1f}{row['asr']:>10.3f}"
            f"{row['precision_at_k']:>14.3f}{pareto:>9}"
        )
    if include_generation:
        print()
        print(f"{'layer':<24}{'gen latency(ms)':>18}{'gen asr':>10}{'faithfulness':>14}")
        for name, row in layers.items():
            gen = row.get("generation")
            if gen is None:
                continue
            gen_asr = "n/a" if gen["generation_asr"] is None else f"{gen['generation_asr']:.3f}"
            faith = (
                "n/a"
                if gen["faithfulness_precision"] is None
                else f"{gen['faithfulness_precision']:.3f}"
            )
            print(
                f"{name:<24}{gen['generation_latency_ms']['mean']:>18.1f}{gen_asr:>10}{faith:>14}"
            )


def run_cost_accuracy_sweep(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    base_policy_config: PolicyConfig = DEFAULT_POLICY_CONFIG,
    generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
    faithfulness_config: FaithfulnessConfig = DEFAULT_FAITHFULNESS_CONFIG,
    asr_sample_size: int | None = 6,
    precision_sample_size: int | None = 10,
    include_generation: bool = False,
    generation_asr_sample_size: int | None = 3,
    generation_precision_sample_size: int | None = 5,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the full leave-one-out defensive-layer ablation and write the
    cost-accuracy frontier report.

    Defaults sample a small subset of the full 20 ASR / 106 precision query
    sets: with detection scoring enabled (four of five layers), a single
    `retrieve()` call costs ~10s (sequential per-candidate perplexity
    scoring -- see PROGRESS.md's Phase 5/8 latency notes), so the full set
    across every layer would take well over an hour. Pass sample sizes >=
    the full query count (or the literal counts, 20/106) to run everything.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_asr_queries, all_precision_queries, n_overlap_excluded = build_query_sets(manifest)
    asr_queries = _sample_queries(all_asr_queries, asr_sample_size, seed)
    precision_queries = _sample_queries(all_precision_queries, precision_sample_size, seed)
    print(
        f"Cost-accuracy sweep: {len(asr_queries)}/{len(all_asr_queries)} ASR queries, "
        f"{len(precision_queries)}/{len(all_precision_queries)} precision queries "
        f"({n_overlap_excluded} overlapping rows already excluded from precision)."
    )

    embedder = Embedder()
    client = get_client()

    layers: dict[str, dict] = {}

    print(f"Running {UNDEFENDED_LABEL} (zero-defense reference)...")
    layers[UNDEFENDED_LABEL] = _run_undefended_baseline(
        asr_queries, precision_queries, embedder, client, base_policy_config
    )

    layer_configs = build_layer_configs(base_policy_config)
    full_defense_result: dict | None = None
    for layer in layer_configs:
        if layer.name == "no_faithfulness_check" and full_defense_result is not None:
            # This toggle is generation-only by construction -- the
            # retrieval layer is byte-for-byte identical to full_defense,
            # so reuse that measurement instead of paying for a duplicate,
            # identical retrieve() sweep.
            layers[layer.name] = dict(full_defense_result)
            layers[layer.name]["note"] = (
                "retrieval layer identical to full_defense -- faithfulness "
                "only applies at generation time, see its generation sub-metrics"
            )
            print(f"Running {layer.name}... (reused full_defense retrieval measurement)")
            continue
        print(f"Running {layer.name} ({len(asr_queries) + len(precision_queries)} queries)...")
        layers[layer.name] = _run_retrieval_layer(
            layer, asr_queries, precision_queries, embedder, client
        )
        if layer.name == FULL_DEFENSE_LABEL:
            full_defense_result = layers[layer.name]

    # Pareto-frontier annotation over the retrieval-level rows.
    rows = list(layers.values())
    for row in rows:
        row["pareto_optimal"] = not _is_dominated(row, rows)

    if include_generation:
        gen_asr_queries = _sample_queries(all_asr_queries, generation_asr_sample_size, seed)
        gen_precision_queries = _sample_queries(
            all_precision_queries, generation_precision_sample_size, seed
        )
        print(
            f"\n--include-generation: {len(gen_asr_queries)} ASR + "
            f"{len(gen_precision_queries)} precision queries per layer, real LLM calls."
        )
        llm = get_llm(temperature=generation_config.generation_temperature)
        faithfulness_checker = FaithfulnessChecker(faithfulness_config)
        for layer in layer_configs:
            print(f"  generation leg: {layer.name}...")
            layers[layer.name]["generation"] = _run_generation_layer(
                layer,
                gen_asr_queries,
                gen_precision_queries,
                embedder,
                client,
                generation_config,
                faithfulness_config,
                llm,
                faithfulness_checker,
            )

    print()
    _print_report(layers, include_generation)

    results = {
        "manifest_path": str(manifest_path),
        "n_chunks": len(manifest),
        "seed": seed,
        "n_asr_queries_used": len(asr_queries),
        "n_precision_queries_used": len(precision_queries),
        "n_asr_queries_total": len(all_asr_queries),
        "n_precision_queries_total": len(all_precision_queries),
        "include_generation": include_generation,
        "base_policy_config": asdict(base_policy_config),
        "layers": layers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--asr-sample-size",
        type=int,
        default=6,
        help="Seeded random subsample of ASR queries (of 20 total). Pass a value >= 20 "
        "to run the full set. Local-only, no API cost -- kept small by default only "
        "because detection-scoring-enabled layers cost ~10s/query.",
    )
    parser.add_argument(
        "--precision-sample-size",
        type=int,
        default=10,
        help="Seeded random subsample of precision queries (of 106 total). Pass a value "
        ">= 106 to run the full set.",
    )
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Also measure the faithfulness_check layer's generation-stage cost/accuracy "
        "(real LLM calls -- needs a working LLM_PROVIDER API key/budget).",
    )
    parser.add_argument("--generation-asr-sample-size", type=int, default=3)
    parser.add_argument("--generation-precision-sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_cost_accuracy_sweep(
        manifest_path=args.manifest,
        output_path=args.output,
        asr_sample_size=None if args.asr_sample_size <= 0 else args.asr_sample_size,
        precision_sample_size=(
            None if args.precision_sample_size <= 0 else args.precision_sample_size
        ),
        include_generation=args.include_generation,
        generation_asr_sample_size=(
            None if args.generation_asr_sample_size <= 0 else args.generation_asr_sample_size
        ),
        generation_precision_sample_size=(
            None
            if args.generation_precision_sample_size <= 0
            else args.generation_precision_sample_size
        ),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
