# VeritasRAG — Progress Log

## Phase 0 — Environment & repo scaffold — DONE

- Scaffolded the repo layout from `plan_2.md` Part 3: `src/{ingestion,embeddings,vectorstore,detection,retrieval,generation,unlearning,observability,api}/`, `dashboard/`, `eval/{benchmarks/}`, `tests/`, `docs/`, `.github/workflows/`. All `src/` modules created as empty stubs (no logic yet), with `__init__.py` in every package dir.
- `pyproject.toml`: hatchling build backend, `requires-python = ">=3.11"`, dependencies from `plan_2.md` Part 1's tech-stack table, `[tool.ruff]` and `[tool.pytest.ini_options]` sections.
- `docker-compose.yml`: single `qdrant/qdrant:latest` service, ports 6333 (REST)/6334 (gRPC), volume `./qdrant_storage`.
- `.gitignore`: standard Python ignores + `qdrant_storage/` + stray WSL `*:Zone.Identifier` files.
- Verified: `docker compose up -d` starts Qdrant; `curl localhost:6333/collections` → `HTTP 200`.

## Phase 1 — Data ingestion + synthetic poisoned corpus — DONE

- `src/ingestion/loaders.py`: `load_text`/`load_pdf`/`load_html` (LangChain `TextLoader`/`PyPDFLoader`/`BSHTMLLoader`), `load_document()` dispatching by extension, `chunk_documents()`/`load_and_chunk()` via `RecursiveCharacterTextSplitter`.
- `eval/synthetic_poison_generator.py`: seeded PoisonedRAG-style synthetic corpus generator. Saves `{chunk_id, text, is_poisoned, target_question, source_doc}` manifest to `eval/benchmarks/poison_manifest.json`. Fully seeded via `random.Random(seed)`.
  - **Dataset deviation (flagging per CLAUDE.md):** uses HF dataset `sentence-transformers/natural-questions` (a lightweight query/gold-passage re-release of Natural Questions) instead of the full `natural_questions` release, which is tens of GB of raw Wikipedia/HTML dumps — impractical for local/student-budget dev. Gives real passage text for clean chunks without the download cost.
  - Clean chunks: sampled passages run through `chunk_documents()` from the Phase 1 loader (reuses the real ingestion splitter rather than a separate ad hoc one).
  - Poisoned chunks: template-crafted adversarial passages that echo the target question and assert a confident, cross-question decoy "fact" (5 template variants, cycled).
- Dependencies installed into `.venv` (Phase-1-relevant subset only, not the full Part 1 stack): `langchain`, `langchain-community`, `langchain-text-splitters`, `pypdf`, `beautifulsoup4`, `lxml`, `datasets`, `pytest`, `fpdf2`, `ruff`, `python-dotenv`. Heavier packages (torch/transformers pipelines, faiss, etc.) deferred to the phase that needs them.
- Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` so `import src...` resolves under pytest.
- Tests: `tests/ingestion/test_loaders.py`, `tests/eval/test_synthetic_poison_generator.py` — 11 tests, all passing (text/HTML/PDF load + chunk, unsupported-extension error, manifest reproducibility, seed-sensitivity, manifest-label sanity check, exit-criteria count check).

### Eval numbers (this session)

- `python eval/synthetic_poison_generator.py` (default params, seed 42) → **280 total chunks: 220 clean, 60 poisoned** (exit criteria: ≥200 clean, ≥50 poisoned — met).
- Reproducibility: two runs with the same seed produced a byte-identical `poison_manifest.json`.
- `pytest tests/`: **11 passed**, 0 failed.
- `ruff check` / `ruff format --check`: clean across `src/`, `eval/`, `tests/`.

## Phase 2 — Embeddings & vector DB with trust-tier schema — DONE

- `src/ingestion/provenance.py`: Ed25519 content-hash signing, entirely through the `cryptography` package (hashing via `hashes.Hash`, not `hashlib`, per `CLAUDE.md`). `ProvenanceSigner` loads/generates one project-wide keypair persisted under `keys/` (now gitignored — private key material, unencrypted PKCS8 PEM, fine for local dev only). `verify_signature` hardened to return `False` (not raise) on malformed/corrupted signature data.
- `src/embeddings/embedder.py`: `Embedder` wraps `sentence-transformers`, default model `BAAI/bge-small-en-v1.5` (384-dim), overridable via `EMBEDDING_MODEL_NAME` env var.
- `src/vectorstore/schema.py`: `TrustTier` enum (`verified`/`unverified`/`quarantined`) + `build_payload()` for the collection's payload fields.
- `src/vectorstore/qdrant_client.py`: collection lifecycle (`ensure_collection`), deterministic UUID5 point IDs (idempotent re-upserts), `upsert_manifest()` batch pipeline, `verify_sample()` (checks signature validity *and* that the stored hash reproduces from the stored text). CLI: `python -m src.vectorstore.qdrant_client upsert|verify`.
- **Design decisions (flagging per `CLAUDE.md`):**
  - `trust_tier` is **not** derived from the Phase 1 manifest's `is_poisoned` ground-truth label — every chunk is upserted as `unverified`. Copying the label in would make the Phase 3 detector eval circular (trivially "detecting" a value it was told). Confirmed live: the Qdrant payload never contains `is_poisoned`.
  - Payload also stores a `text` field beyond the 7 schema fields listed in the phase prompt (`doc_id`, `chunk_id`, `source_uri`, `trust_tier`, `provenance_signature`, `ingested_at`, `content_hash`) — without chunk text, later retrieval/generation phases would have nothing to feed the LLM.
- Dependencies newly installed into `.venv`: `qdrant-client`, `cryptography`, `sentence-transformers` (pulls `torch`, `transformers`). Hit `TMPDIR`/disk-space issue installing `sentence-transformers` (see below), resolved by pointing `TMPDIR` at the real disk instead of the small `/tmp` tmpfs.
- Tests: `tests/ingestion/test_provenance.py`, `tests/embeddings/test_embedder.py`, `tests/vectorstore/test_schema.py`, `tests/vectorstore/test_qdrant_client.py` (real integration test against `localhost:6333` using a disposable scratch collection, auto-skips if Qdrant isn't reachable) — 16 new tests, all passing.

### Eval numbers (this session)

- `python -m src.vectorstore.qdrant_client upsert` → **280/280** Phase 1 chunks embedded, signed, and upserted into collection `veritasrag_chunks` (384-dim, cosine distance).
- `python -m src.vectorstore.qdrant_client verify --sample-size 8` → **8/8** sampled signatures valid, content-hash-matches-text confirmed for all.
- `pytest tests/`: **27 passed**, 0 failed (11 from Phase 1 + 16 new).
- `ruff check` / `ruff format --check`: clean across `src/`, `eval/`, `tests/`.

### Infra note

- In this WSL2 environment, `/tmp` is a 1.9GB tmpfs; pip's default download temp dir lives there and a large wheel (`torch`, 423MB) can exhaust it mid-download (`[Errno 28] No space left on device`) even though the real disk (`/dev/sdd`) has ~950GB free. Fix: set `TMPDIR` to a directory on the real disk (used `/home/lenovo/.pip_tmp`) before `pip install` for any future heavy package installs (e.g. Phase 3's `faiss-cpu`/`transformers` pipelines).

## Phase 3 — Poison detection module — DONE

Four detection signals, fused into a single per-chunk `poison_score`, plus a standalone LLM-as-judge baseline for comparison:

- `src/detection/config.py`: new centralized `DetectionConfig` frozen dataclass (first shared config module in the repo, vs. the existing per-module `os.getenv` pattern) — every threshold/weight/model-name used across the detection module lives here, `os.getenv`-backed via `field(default_factory=...)` so each instantiation re-reads the environment (a plain `os.getenv(...)` default expression would only evaluate once at class-definition time — hit this as a real test failure, fixed). Every detection function takes an optional `config: DetectionConfig` param instead of reaching for a global, so Phase 9's cost-accuracy sweep can construct alternate config instances directly. `OPENAI_API_KEY` deliberately never lives on this dataclass (it gets serialized into eval-result JSON).
- `src/detection/perplexity.py`: `PerplexityScorer` wraps a small HF causal LM, default `distilgpt2` (not `gpt2` — empirically the stronger/faster choice on this CPU-only box). `normalize_perplexity()` uses fixed config-driven anchors, not batch min-max, so scores stay comparable across batches of different size/composition.
- `src/detection/cluster_anomaly.py`: combines a faiss k-NN density signal with a `DBSCAN` small-cluster-membership signal. Standard `LocalOutlierFactor`/`IsolationForest` were benchmarked and explicitly rejected — both scored *below* the 21% base rate on this manifest, because they hunt for sparse-region outliers while this data's actual poison pattern is dense micro-clusters (near-duplicate PoisonedRAG variants targeting the same question) inside a diffuse background.
- `src/detection/classifier.py`: lightweight `StandardScaler` + `LogisticRegression(class_weight="balanced")` on chunk embeddings stacked with normalized perplexity/cluster scores — not a fine-tuned HF transformer (only 280 labeled examples, 60 positive, CPU-only box). Evaluated via stratified 5-fold CV with `cross_val_predict` rather than a single 80/20 split (60 positives make a single split's held-out metric too high-variance); a separate model fit on the full manifest is what gets persisted to `models/poison_classifier.joblib` for production scoring, but its in-sample predictions are never reported as a metric.
- `src/detection/pipeline.py`: `DetectionPipeline` class fuses the three signals via `fuse_scores()`, a pure function that renormalizes weights when the classifier is unavailable (it's genuinely optional). The LLM-judge is deliberately *not* a fusion input — it's a standalone comparison baseline per the exit-criteria wording.
- `src/detection/llm_judge.py`: OpenAI-based judge (per `PLAN.md`'s literal Phase 3 wording — this is *not* a deviation from `PLAN.md` itself, only from the superseded `plan_2.md` and `pyproject.toml`'s prior Anthropic-only stance; new `openai` dependency added). Structured JSON output (`poison_likelihood` float, not an ambiguous separate confidence+bool pair), `ThreadPoolExecutor`-parallelized, with an on-disk cache (`eval/benchmarks/.judge_cache.json`, gitignored) to avoid re-spending API budget across repeated dev runs.
- `eval/run_eval.py`: `--stage detection` CLI. Computes embeddings/perplexity/cluster-anomaly once and reuses them across the classifier and fusion steps. Critically, the "fused pipeline" report row uses the classifier's **CV out-of-fold** scores, not `DetectionPipeline`'s live full-fit-model prediction — using the latter would leak the classifier's own training data into the fused score's reported metric.
- **Real bug caught during verification**: training via `python -m src.detection.classifier train` pickled `ClassifierArtifact` with its module recorded as `__main__` (since that file was the entry point), which then failed to unpickle from a different entry point (`eval/run_eval.py`) with `AttributeError: module '__main__' has no attribute 'ClassifierArtifact'`. Fixed by persisting a plain dict of the artifact's fields instead of the dataclass instance.
- Tests: `tests/detection/{test_config,test_perplexity,test_cluster_anomaly,test_classifier,test_pipeline,test_llm_judge}.py`, 21 tests (19 run locally + deterministically; `test_llm_judge.py`'s 2 tests skip via `pytest.skip(...)` in a fixture when `OPENAI_API_KEY` is unset, mirroring the existing Qdrant-unreachable skip pattern).
- New deps installed: `faiss-cpu`, `openai`, `joblib` (scikit-learn was already installed). New files: `.env.example` (didn't exist before despite `.gitignore` already carrying the exception for it).

### Eval numbers (this session, `python eval/run_eval.py --stage detection`, full run incl. LLM-judge, 280 chunks: 220 clean / 60 poisoned)

| method | precision | recall | F1 |
|---|---|---|---|
| perplexity alone | 0.429 | 0.800 | 0.558 |
| cluster-anomaly alone | 0.247 | 0.667 | 0.360 |
| classifier (5-fold CV, out-of-fold) | 0.918 | 0.933 | 0.926 |
| fused pipeline | 0.918 | 0.933 | 0.926 |
| **LLM-judge (OpenAI, `gpt-5.4-mini`)** | **0.968** | **1.000** | **0.984** |

Note on the fusion weights: the first run with the originally-planned default weights (perplexity 0.4 / cluster 0.25 / classifier 0.35) produced a *worse* fused F1 (0.862) than the classifier alone (0.926) — diluting the classifier's output (which already stacks perplexity+cluster as features) with the much weaker standalone signals actively hurt it. Recalibrated the defaults to perplexity 0.25 / cluster 0.15 / classifier 0.60 based on this run's own numbers, which brought the fused pipeline back up to match the classifier's CV performance exactly. This is a coarse one-run manual calibration, not a proper sweep — real tuning is Phase 9's job, which is exactly why these are config fields (`FUSION_*_WEIGHT` env vars) and not hardcoded.

**The zero-shot LLM-judge baseline outright beat the combined local detector** (0.984 vs 0.926 F1, missing zero poisoned chunks). This is a fair, non-circular comparison — the judge got no ground-truth labels, just the raw chunk text and a description of what a PoisonedRAG-style passage looks like. Worth remembering for Phase 4/9 framing: the cheap local pipeline (no per-call API cost, no external dependency) is respectable but not yet competitive with a frontier model as judge on this synthetic corpus; the honest story is a cost/latency-vs-accuracy tradeoff, not "our detector wins."

**Also discovered, uncommitted in the working tree (not made by me this session)**: `PLAN.md` and `plan_2.md` show as locally modified against the last commit. Notably, the committed `PLAN.md` says the Phase 3 judge baseline should use "the Claude API," while the working-tree version (what this session's prompt was copied from) says "the open_ai API" — i.e. that wording change already existed in the working tree before this session started, uncommitted. Flagged for visibility; left untouched since it looks like your own in-progress edit.

## Outstanding / next up

- Full Part 1 dependency stack (`ragas`, `opentelemetry`, `streamlit`, etc.) still only declared in `pyproject.toml`, not installed — install incrementally per phase as needed.
- `.env` now has `OPENAI_API_KEY` set (added mid-session, used for the LLM-judge run above) — but still no `ANTHROPIC_API_KEY`, which will be needed from Phase 5 (generation) onward.
- `keys/` (Ed25519 provenance keypair) and `models/` (trained poison classifier) are both gitignored and local-only — anyone else running this repo will regenerate/retrain their own on first run.
- Phase 4 — Provenance-gated retrieval policy engine (not started). Uses Plan Mode per `PLAN.md`. Will consume `DetectionPipeline.score_chunks()`'s `poison_score` directly; note `cluster_anomaly.py`'s batch-relative normalization is a known limitation there (a 5-10 candidate per-query batch has a different reference distribution than this phase's 280-chunk eval batch) — `score_cluster_anomaly`'s `reference_embeddings` param already supports a disjoint reference/query call to help with this, but the DBSCAN sub-signal is documented as self-referential-only for now.
