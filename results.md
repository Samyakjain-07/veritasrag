# VeritasRAG — Results

**Status as of this report:** Phases 0–5 of `PLAN.md` are complete (`PROGRESS.md`); Phase 6 (verifiable unlearning), Phase 7 (observability), and Phase 8 (agentic/multi-tenancy) have not been started. Every number below is pulled directly from committed eval artifacts (`eval/benchmarks/*.json`), the codebase, or a deterministic re-run of already-implemented Phase 0–5 code (see Stage 3). **No number in this report is estimated, extrapolated, or fabricated.** Where the underlying implementation or eval harness doesn't exist yet, the cell says so explicitly and names what would need to be built, rather than guessing.

---

## Stage 1 — Result Tables

### Table I — Corpus / Dataset Statistics

| Field | Value |
|---|---|
| Total chunks | 280 |
| Clean chunks | 220 |
| Synthetic poisoned chunks | 60 |
| Poison ratio | 21.43% (60/280) |
| Attack template family | PoisonedRAG-style, 5 template variants cycled (`eval/synthetic_poison_generator.py::POISON_TEMPLATES`) |
| # poison target rows / ASR queries | 20 |
| # clean-precision queries | 106 (16 rows excluded — a poison-target row whose source doc also produced a clean chunk; excluded only from the precision set so precision loss isn't confounded with the attack itself, see `build_query_sets` docstring) |
| Underlying real-passage source | HF `sentence-transformers/natural-questions` (query/gold-passage re-release), 150 sampled questions, split=`train`, seed=42 |
| Chunking | `RecursiveCharacterTextSplitter`, chunk_size=500, chunk_overlap=50 chars (`src/ingestion/loaders.py`) |
| Train / dev / test splits | **None exist.** The detector is evaluated via 5-fold stratified CV (`cross_val_predict`) over the full 280-chunk corpus, not a fixed split. Retrieval/generation are evaluated against the full corpus with the 20-query ASR set / 106-query precision set above. If the paper needs a held-out test split, that's new eval-harness work, not a report gap. |
| Embedding model | `BAAI/bge-small-en-v1.5` (`sentence-transformers`), 384 dimensions, cosine-normalized |
| Vector-DB backend | Qdrant `1.18.0` (Docker, `qdrant/qdrant:latest`), collection `veritasrag_chunks`, cosine distance |

---

### Table II — Main Result (ASR reduction)

| Configuration | ASR ↓ | ASR Δ (pp) | Clean ACC ↑ | Retrieval Precision@k ↑ | Recall@5 ↑ |
|---|---|---|---|---|---|
| Undefended baseline | **1.000** | — | Not measured | 0.345 | Not measured |
| + Layer 1 (poison detection) only | *Not isolated by current eval* | — | — | — | — |
| + Layer 2 (provenance/trust-tier gating) only | *Not isolated by current eval* | — | — | — | — |
| + Layer 1 + Layer 2 combined (current `defended` run) | **0.000** | **−100.0** | Not measured | 0.347 | Not measured |
| + Layer 3 (unlearning) | *Not implemented* | — | — | — | — |
| Full VeritasRAG (all layers) | *Not implemented* | — | — | — | — |

**Gaps, explained:**
- **Layer 1 vs. Layer 2 isolation**: `src/retrieval/policy_engine.py::_gate()` reads the fused `poison_score` (Layer 1's output) *and* `trust_tier` (Layer 2's input) together in one decision function — there's no eval mode that runs one without the other. Isolating them needs a new `--stage retrieval --ablate {detection,gating}` flag in `eval/run_eval.py` that either force-sets `poison_score=0` (gating-only) or force-sets every `trust_tier` to a fixed value (detection-only).
- **Layer 3 / Full VeritasRAG**: Phase 6 (`src/unlearning/{delete,receipt,ghost_vector_defense}.py`) is entirely unimplemented (0-byte stub files) — there is no unlearning layer to turn on.
- **Clean ACC↑**: no answer-correctness eval exists. The closest measured proxy is `faithfulness_precision = 0.425` (Phase 5, generation stage) — but that metric conflates correct abstention with hallucination (documented limitation, see Stage 2) and is not the same thing as accuracy, so it is not substituted here.
- **Recall@5**: `eval/run_eval.py::_precision_at_k` only computes precision (fraction of the top-k that are relevant); no recall-at-k is currently computed anywhere in the codebase.
- **Exit criteria (the one row that *is* measured):** ASR reduction ≥30pp → **100.0pp, PASSED**. Retrieval-precision loss <15% (relative) → **−0.55% (i.e. precision improved, not lost), PASSED**. Source: `eval/benchmarks/retrieval_eval_results.json::exit_criteria.passed = true`.

---

### Table III — Poison-Detector Metrics (per method, single attack type/poison rate)

Only one attack type (the PoisonedRAG-style template family above) and one poison rate (21.43%) currently exist in the corpus — there is no per-attack-type or per-poison-rate sweep yet (see Table III gap note below). TPR/FPR/Clean Retention Rate are not persisted directly; they are exactly reconstructed here from the stored precision/recall and the known 60-poisoned/220-clean split (verified: for perplexity, cluster-anomaly, and classifier/fused, the reconstructed TP/FP counts reproduce the stored precision/recall to full float64 precision).

| Method | Precision | Recall (=TPR) | F1 | FPR | Clean Retention Rate |
|---|---|---|---|---|---|
| Perplexity alone | 0.4286 | 0.8000 | 0.5581 | 0.2909 | 0.7091 |
| Cluster-anomaly alone | 0.2469 | 0.6667 | 0.3604 | 0.5545 | 0.4455 |
| Classifier (5-fold CV, out-of-fold) | 0.9180 | 0.9333 | 0.9256 | 0.0227 | 0.9773 |
| Fused pipeline (perplexity 0.25 / cluster 0.15 / classifier 0.60) | 0.9180 | 0.9333 | 0.9256 | 0.0227 | 0.9773 |
| LLM-judge (OpenAI `gpt-5.4-mini`, zero-shot, no ground truth) | 0.9677* | 1.0000* | 0.9836* | 0.0091* | 0.9909* |

\* **Reproducibility caveat**: the LLM-judge row is *not* in the machine-readable `eval/benchmarks/detection_eval_results.json` (that file's `metrics` dict has no `llm_judge` key — it was produced by a run without `--include-judge`, which skips the paid OpenAI calls). These numbers are reconstructed from `PROGRESS.md`'s narrative log of the Phase 3 session (precision 0.968, recall 1.000, F1 0.984, rounded to 3dp) assuming integer TP/FP counts consistent with those rounded values (TP=60, FP=2) — the exact fractions were verified to round correctly, but there is no persisted raw judge-score artifact to regenerate this row deterministically without re-spending OpenAI API budget. If this number needs to be citable without caveat, re-run `python eval/run_eval.py --stage detection --include-judge` and use `eval/benchmarks/detection_eval_results.json` directly.

**Gap**: per-attack-type and per-poison-rate breakdown does not exist — the corpus generator (`eval/synthetic_poison_generator.py`) produces one attack family at one fixed poison ratio per run. A sweep would mean generating several manifests at different `--num-poison-targets`/`--poison-per-target` values (and, for attack-type breadth, implementing additional non-PoisonedRAG-style attack templates) and re-running detection eval on each.

---

### Table IV — Ablation (RAGuard-style layer on/off)

| Layer 1 (detection) | Layer 2 (gating) | Layer 3 (unlearning) | ASR ↓ | FPR ↓ |
|---|---|---|---|---|
| off | off | off | 1.000 | — |
| on | on | off | 0.000 | — |
| on | off | off | *Not isolated* | — |
| off | on | off | *Not isolated* | — |
| any | any | on | *Not implemented (Phase 6)* | — |

Same root cause as Table II: Layers 1 and 2 are architecturally fused in `policy_engine.py::_gate()` (not a bug — Phase 4's design deliberately makes trust-tier and poison-score compound rather than gate independently), so today's eval harness only has two real data points: all-off (undefended) and 1+2-combined-on (defended). The "FPR↓" column as a *retrieval-level* ablation metric (fraction of clean chunks wrongly gated out, per combination) isn't logged anywhere — the closest existing number is Table III's *detection-level* FPR (0.0227 for the fused pipeline), which is a different, upstream quantity. Producing a real Table IV needs a new `eval/run_eval.py --stage ablation` entry point that runs `retrieve()` under each of the 2³ (or, honestly, 2² until Layer 3 exists) on/off combinations and logs both ASR and a retrieval-level clean-chunk-drop rate per combination.

---

### Table V — Provenance / Trust-Tier Gating

| Trust tier | Poison-score downweight threshold | Poison-score hard-drop threshold | Chunks at this tier in current corpus |
|---|---|---|---|
| `verified` | 0.55 | 0.85 | **0** |
| `unverified` | 0.35 | 0.65 | **280** (100%) |
| `quarantined` | n/a (categorical drop unless `allow_quarantined_tier=True`) | n/a | **0** |

(Thresholds from `PolicyConfig`, `eval/benchmarks/retrieval_eval_results.json::config`.)

**Gap, explained**: Phase 2's ingestion pipeline deliberately upserts *every* chunk as `trust_tier=unverified` (`PROGRESS.md` Phase 2 note: copying the ground-truth poison label into `trust_tier` would make the Phase 3 detector eval circular). As a direct consequence, the `verified` and `quarantined` tiers — and their distinct, more/less permissive poison-score bands — are implemented and unit-tested (`tests/retrieval/test_policy_engine.py`) but **never exercised by real corpus data**. The only real, measured tier-gating outcome is: under the `unverified` band, all 20/20 targeting poison chunks were gated out of the top-k (ASR 1.000→0.000) with retrieval precision unchanged within noise (0.345→0.347). Exact per-query counts of *how many clean docs were gated out* aren't persisted by the current eval (it logs aggregate precision@k, not a drop ledger) — that near-zero precision delta upper-bounds the clean-doc loss at effectively none, but isn't an exact count. A real Table V needs (a) a corpus with chunks actually signed/labeled at `verified` and `quarantined` tiers, and (b) the eval to log a per-query gated-chunk ledger instead of just aggregate precision.

---

### Table VI — Verifiable Unlearning

**Not implemented.** `src/unlearning/delete.py`, `src/unlearning/receipt.py`, and `src/unlearning/ghost_vector_defense.py` are all present as 0-byte stub files from the Phase 0 scaffold — Phase 6 of `PLAN.md` has not been started. None of the following exist yet: deletion-by-ID against Qdrant, signed deletion receipts, ghost-vector reconstruction defense, or an unlearning eval stage (`eval/run_eval.py --stage unlearning` is referenced in `PLAN.md`'s Phase 6 verify step but doesn't exist in the current `run_eval.py`).

| Metric | Value |
|---|---|
| Deletion latency (total / per-record) | Not implemented |
| Residual retrievability / removal-precision@k, before→after | Not implemented |
| Membership-inference / re-retrieval check | Not implemented |
| Receipt-generation time | Not implemented |
| Receipt-verification time | Not implemented |
| Receipt validity (%) | Not implemented |

---

### Table VII — Latency / Overhead

| Stage | Mean (ms) | p50 (ms) | p95 (ms) | Sample size |
|---|---|---|---|---|
| Retrieval leg (embed query → Qdrant search → `DetectionPipeline` live-score → gate) | 9937.8 | 9783.8 | 10932.6 | n=10 (ASR sample) / n=20 (precision sample), same underlying measurement |
| Generation leg (LLM call + NLI faithfulness check) | 2907.3 | 2260.7 | 5430.6 | n=10 |
| End-to-end (`/query`) | 12845.1 | 12478.6 | 15344.7 | n=10 |

Source: `eval/benchmarks/generation_eval_results.json::metrics.latency_ms`.

**Gaps, explained**: there is no logged "undefended baseline query latency" (a raw Qdrant top-k search with no detection/gating) to diff against — the retrieval-stage eval logs ASR/precision, not timing, for the undefended path. There is also no per-layer latency breakdown (detection-signal-scoring time vs. gating-decision time vs. Qdrant-search time) — the 9937.8ms retrieval-leg figure is the *combined* Layer1+Layer2 latency, dominated (per `PROGRESS.md`'s Phase 5 note) by `DetectionPipeline`'s sequential, non-batched per-candidate perplexity scoring, a known Phase-9-cost-sweep target rather than something already broken out into separate numbers. Getting a true "baseline vs. added latency per layer" table needs new instrumentation (e.g. `time.perf_counter()` spans around each sub-step in `policy_engine.py::retrieve()`, or the OpenTelemetry spans that Phase 7 — also not started — is supposed to add).

---

## Stage 2 — Evaluation Metrics: Formulas & Measurement Code Paths

### Attack Success Rate (ASR)

**Definition (substring/identity-of-source matching, per PoisonedRAG's convention** — Zou et al., *"PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models,"* 2024**):** an attack against a target query succeeds if at least one of the attacker's poisoned passages targeting that query is present in the system's final retrieved/answer-grounding set. PoisonedRAG's own paper measures this at the generation level via substring-matching the attacker's injected target string in the generated answer; this repo measures it at two levels using an equivalent identity check (a poisoned chunk *is* the injected content, so "chunk ID present in the result set" and "target string present in the output" are the same event by construction, since only the poisoned chunk's text can produce that string):

- **Retrieval-level ASR** (Table II's undefended/defended rows):
  $$\text{ASR}_{\text{retrieval}} = \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}\left[\exists\, c \in \text{topk}(q) : c \in \text{PoisonChunks}(q)\right]$$
  Code: `eval/run_eval.py::_attack_succeeded(chunk_ids, poison_chunk_ids)` — `True` iff any returned `chunk_id` is in the query's poison-chunk set; aggregated in `run_retrieval_eval` (`eval/run_eval.py:319`).
- **Generation-level ASR** (Table II's `generation_asr`, stricter bar): an attack only counts as succeeding if a generated answer *claim* is NLI-entailed by a poisoned chunk, not merely retrieved. Code: `eval/run_eval.py::run_generation_eval` (`eval/run_eval.py:480`), using `src/generation/faithfulness.py::FaithfulnessChecker` for the entailment check.

### Retrieval Precision@k

$$\text{Precision@k}(q) = \frac{|\text{topk}(q) \cap \text{Relevant}(q)|}{|\text{topk}(q)|}, \quad \text{Precision@k} = \frac{1}{|Q|}\sum_{q} \text{Precision@k}(q)$$

An empty result set scores 0.0 (not an excluded/vacuous 1.0). Code: `eval/run_eval.py::_precision_at_k` (`eval/run_eval.py:275`).

### Detector Precision / Recall / F1

Standard binary classification metrics against the Phase 1 manifest's `is_poisoned` ground truth. Code: `eval/run_eval.py::_binary_metrics`, wrapping `sklearn.metrics.precision_recall_fscore_support(average="binary")` (`eval/run_eval.py:67`).

### Detector TPR / FPR / Clean Retention Rate

$$\text{TPR} = \text{Recall} = \frac{TP}{TP+FN}, \qquad \text{FPR} = \frac{FP}{FP+TN}, \qquad \text{Clean Retention Rate} = 1 - \text{FPR} = \frac{TN}{FP+TN}$$

Not directly computed anywhere in the codebase today (Table III's values were hand-derived from the stored precision/recall + known class counts, see Table III's note). A reproducible code path would extend `_binary_metrics` to also return a confusion matrix via `sklearn.metrics.confusion_matrix(y_true, y_pred)`.

### Faithfulness Precision

Fraction of generated answer claims (via `src/generation/faithfulness.py::split_claims`, regex sentence splitting with a citation-fragment filter) whose best cross-encoder NLI entailment probability against any context chunk clears `faithfulness_entailment_threshold` (default 0.5). Code: `src/generation/faithfulness.py::FaithfulnessChecker`, aggregated in `eval/run_eval.py::run_generation_eval`. **Documented limitation**: conflates correct abstention (the system saying "I don't have enough information") with hallucination, since an abstention sentence also gets NLI-scored against context and typically finds no entailment — this metric is a lower bound on true groundedness, not a hallucination rate (see `run_generation_eval`'s docstring and `PROGRESS.md` Phase 5).

### Latency (mean / p50 / p95)

Wall-clock `time.perf_counter()` deltas around each pipeline leg, collected per query and reduced via `eval/run_eval.py::_percentile` (linear-interpolation percentile, `eval/run_eval.py:463`) for p50/p95, `statistics.mean`-equivalent for mean.

### ROC-AUC / DET curve (detector)

Standard `sklearn.metrics.roc_curve`/`det_curve`/`roc_auc_score` over the classifier's continuous out-of-fold probability (`predict_proba`, positive class) vs. ground-truth labels. Code (new, Stage 3): `eval/export_figures.py::export_detector_roc_det`, consuming `models/poison_classifier.joblib`'s persisted `cv_predictions` (produced by `src/detection/classifier.py::train_classifier`'s `cross_val_predict(..., method="predict_proba")`, `src/detection/classifier.py:98`).

---

## Stage 3 — Artifacts

### Config block (seeds, hardware, versions, hyperparameters)

```yaml
# Reproducibility / config snapshot — generated from repo state, no invented values.

seeds:
  corpus_generation_seed: 42        # eval/synthetic_poison_generator.py DEFAULT_SEED
  classifier_cv_seed: 42            # src/detection/config.py DetectionConfig.classifier_seed

hardware:
  cpu: "13th Gen Intel(R) Core(TM) i5-1335U"
  cores: 12
  ram_total: "3.7Gi"
  gpu: none                          # torch.cuda.is_available() == False; all models run CPU-only
  os: "Linux (WSL2)"

software_versions:
  python: "3.14.4"
  qdrant-client: "1.18.0"
  qdrant_server_image: "qdrant/qdrant:latest"
  sentence-transformers: "5.6.1"
  torch: "2.13.0"
  transformers: "5.14.1"
  scikit-learn: "1.9.0"
  langchain: "1.3.14"
  openai_sdk: "2.52.0"

models:
  embedding_model: "BAAI/bge-small-en-v1.5"   # 384-dim, cosine-normalized
  perplexity_model: "distilgpt2"
  nli_faithfulness_model: "cross-encoder/nli-deberta-v3-xsmall"
  llm_judge_model: "gpt-5.4-mini"             # OpenAI, detection-stage baseline only
  generation_llm: "config-driven via get_llm(), LLM_PROVIDER=openai default"  # src/generation/llm_client.py

retrieval:
  top_k: 5                          # RETRIEVAL_TOP_K
  overfetch_factor: 3.0             # RETRIEVAL_OVERFETCH_FACTOR

detection_hyperparameters:
  perplexity_threshold: 60.0
  perplexity_low_anchor: 20.0
  perplexity_high_anchor: 120.0
  cluster_knn_k: 5
  cluster_dbscan_eps: 0.25
  cluster_dbscan_min_samples: 2
  cluster_anomaly_threshold: 0.5
  classifier_n_splits: 5
  classifier_threshold: 0.5
  fusion_weights: {perplexity: 0.25, cluster: 0.15, classifier: 0.60}
  poison_score_threshold: 0.5

gating_hyperparameters:
  poison_downweight_threshold_verified: 0.55
  poison_hard_drop_threshold_verified: 0.85
  poison_downweight_threshold_unverified: 0.35
  poison_hard_drop_threshold_unverified: 0.65
  downweight_floor_multiplier: 0.2
  frequency_zscore_threshold: 3.0
  frequency_min_samples: 20
  allow_quarantined_tier: false

ingestion:
  chunk_size: 500
  chunk_overlap: 50

corpus:
  total_chunks: 280
  clean_chunks: 220
  poisoned_chunks: 60
```

### Figures exported this session

Generated by `eval/export_figures.py` (new script, added for this report), both derived from real, already-computed artifacts — not new modeling:

1. **`eval/benchmarks/figures/perplexity_distribution.png`** — raw `distilgpt2` perplexity per chunk, clean vs. poisoned, over all 280 chunks (à la TrustRAG). Recomputed live (deterministic causal-LM forward pass, no sampling): clean mean/median = **54.24 / 44.87**, poisoned mean/median = **101.75 / 95.67**, standalone perplexity-signal ROC-AUC = **0.844**.
   - **Reproducibility note**: this re-run's ROC-AUC (0.844) matches `src/detection/perplexity.py`'s module docstring (0.844) exactly, confirming the same discriminative separation — but the raw mean/median values differ from the docstring's cached numbers (37.6/31.1 clean, 99.9/96.5 poisoned). A `transformers`-library warning surfaced during this run (`loss_type=None was set in the config but it is unrecognized. Using the default loss: ForCausalLMLoss`) suggests the installed `transformers==5.14.1` computes the loss slightly differently than whatever version was active when Phase 3's docstring numbers were written, shifting absolute perplexity values while preserving rank order (hence identical AUC). Flagging rather than silently reconciling — if this discrepancy matters for the paper, worth pinning down which `transformers` version produced the original docstring numbers.
   - Raw per-chunk scores: `eval/benchmarks/perplexity_raw_scores.json`.
2. **`eval/benchmarks/figures/detector_roc_det.png`** — ROC and DET curves for the classifier's 5-fold CV out-of-fold probabilities (`models/poison_classifier.joblib::cv_predictions`), n=280, 60 poisoned. ROC-AUC = **0.992**.

**Not exported — ASR-vs-poison-rate curve**: requires corpora generated at multiple poison ratios (e.g. sweeping `--num-poison-targets`/`--poison-per-target` in `eval/synthetic_poison_generator.py` and re-running `--stage retrieval` at each), which don't exist yet. Only a single ~21.4% poison-rate data point is currently available (a single point can't form a curve).

### Reproduction commands

```bash
# Detection metrics (Table III)
python eval/run_eval.py --stage detection                    # local signals + fused pipeline
python eval/run_eval.py --stage detection --include-judge     # + LLM-judge (costs OpenAI API budget)

# Retrieval / ASR (Table II)
python eval/run_eval.py --stage retrieval --compare-baseline --top-k 5

# Generation / latency (Table VII)
python eval/run_eval.py --full-report --generation-asr-sample-size 10 --generation-precision-sample-size 20

# Figures (Stage 3)
python eval/export_figures.py
```
