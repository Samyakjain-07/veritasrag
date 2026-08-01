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
