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
