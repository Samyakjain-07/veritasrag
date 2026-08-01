# VeritasRAG — Complete Handbook
### Commands, Claude Code prompts, and phases for setup, build, and deployment

This is the single reference document: every terminal command you'll actually type, every Claude Code prompt for every phase, and the deployment steps at the end. Pair it with `CLAUDE.md` (unchanged from before — still drop it in your repo root as-is). This handbook supersedes the earlier phase document; the phases are the same, but every step now has the exact command next to it.

---

## 0. How to use this with Claude Code

One phase per session. At the start of each: *"Read PLAN.md and CLAUDE.md, we're starting Phase N."* Use **Plan Mode** for phases marked so below. At the end of each session: *"Update PROGRESS.md with what's done, what's outstanding, and today's numbers."* Commit and, where noted, tag.

---

## 1. Tech stack (quick reference)

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph / LangChain 1.x |
| Vector DB | Qdrant (Docker) |
| Embeddings | HF `sentence-transformers` |
| Generation | LLM provider (OpenAI by default, swappable via LLM_PROVIDER) |
| Backend | FastAPI |
| Detection | perplexity (small HF causal LM) + scikit-learn/faiss cluster anomaly + optional HF classifier |
| Faithfulness | HF NLI cross-encoder |
| Crypto | Python `cryptography` (Ed25519) — never hand-rolled |
| Eval | Ragas / Phoenix + custom scripts |
| Observability | OpenTelemetry |
| Dashboard | Streamlit → Next.js later if time allows |
| Containers | Docker + Docker Compose |
| CI | GitHub Actions |
| Datasets | Natural Questions, HotpotQA, TOFU (via HF `datasets`) |

Architecture recap: ingestion gateway → embeddings/Qdrant (with trust-tier metadata) → retrieval policy engine (trust gating + anomaly checks) → generation (with citation-faithfulness check) → answer. Unlearning service and observability sit off the critical path, both wired directly into Qdrant and the policy engine. (Full diagram was rendered earlier in this conversation.)

---

## 2. One-time environment setup

Run this once, before Phase 0.

### 2.1 Install Claude Code

```bash
# macOS / Linux / WSL
curl -fsSL https://claude.ai/install.sh | bash
```
```powershell
# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```

Verify:
```bash
claude --version
claude doctor
```

### 2.2 Install the VS Code extension

Open VS Code → `Cmd/Ctrl+Shift+X` → search "Claude Code" → Install. (Or open the project and let Claude Code auto-prompt install when you run `claude` in the integrated terminal.)

### 2.3 Authenticate

```bash
claude
```
Follow the browser prompt. Needs a Claude Pro/Max/Team/Enterprise or Console account — the free claude.ai plan doesn't include Claude Code.

### 2.4 Docker / WSL2

Install Docker Desktop manually (docker.com) — this step can't be scripted. On Windows, enable WSL2 first if you haven't:
```powershell
wsl --install
```

### 2.5 Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2.6 Git + GitHub (keep the repo private for now)

```bash
git init
gh repo create veritasrag --private --source=. --remote=origin
git add -A && git commit -m "chore: initial commit"
git push -u origin main
```

### 2.7 Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
```

### 2.8 Environment variables

Create `.env` in the repo root (already gitignored):
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
QDRANT_URL=http://localhost:6333
```
`LLM_PROVIDER` selects which key `src/generation/llm_client.py` uses ("openai" or "anthropic") — see the full `.env.example` for every overridable var.

### 2.9 Drop in the project files

Place `PLAN.md` (this handbook), `CLAUDE.md`, and an empty `PROGRESS.md` in the repo root:
```bash
touch PROGRESS.md
```

Setup's done. Open Claude Code in the repo and start Phase 0.

---

## 3. Phase-by-phase: implementation & build

### Phase 0 — Repo scaffold
**Builds on:** setup above.

**Claude Code prompt**
```
Set up the repository structure for VeritasRAG as described in PLAN.md
Part 3 [paste the folder tree from the earlier handbook, or ask Claude Code
to propose one matching this phase's components: ingestion, embeddings,
vectorstore, detection, retrieval, generation, unlearning, observability,
api, dashboard, eval, tests]. Create pyproject.toml with the dependencies
in the Tech Stack table, and docker-compose.yml running Qdrant. Scaffolding
only, no business logic.
```

**Verify**
```bash
docker compose up -d
curl localhost:6333/collections
```

**Checkpoint**
```bash
git add -A && git commit -m "chore: scaffold repo, qdrant compose"
git tag v0.0-scaffold
```

---

### Phase 1 — Data ingestion + synthetic poisoned corpus

**Claude Code prompt**
```
Implement Phase 1: src/ingestion/loaders.py for text/PDF/HTML using
LangChain document loaders, and eval/synthetic_poison_generator.py that
pulls a subset of Natural Questions from Hugging Face datasets, generates
PoisonedRAG-style adversarial passages, injects them, and saves a JSON
manifest of {chunk_id, text, is_poisoned, target_question, source_doc}.
Seed all randomness. Write pytest tests including a manifest-label sanity
check.
```

**Verify**
```bash
pytest tests/ -k ingestion
python eval/synthetic_poison_generator.py --seed 42
```

**Exit criteria:** ≥200 clean + ≥50 poisoned chunks; identical output on repeat runs with the same seed.

---

### Phase 2 — Embeddings & vector DB schema

**Claude Code prompt**
```
Implement Phase 2: src/embeddings/embedder.py wrapping a
sentence-transformers model, src/vectorstore/schema.py + qdrant_client.py
for a collection with payload fields {doc_id, chunk_id, source_uri,
trust_tier, provenance_signature, ingested_at, content_hash}, and
src/ingestion/provenance.py signing each chunk's content hash with Ed25519
via the `cryptography` package. Batch-upsert the Phase 1 corpus and write a
script that verifies a sampled chunk's signature.
```

**Verify**
```bash
python -m src.vectorstore.upsert_corpus
python scripts/verify_signature_sample.py
```

---

### Phase 3 — Poison detection module *(use Plan Mode)*

**Claude Code prompt**
```
Implement Phase 3: perplexity scoring (src/detection/perplexity.py),
embedding-space cluster anomaly detection (src/detection/cluster_anomaly.py),
an optional fine-tuned classifier (src/detection/classifier.py), fused into
a poison_score per chunk in src/detection/pipeline.py. Add an LLM-as-judge
baseline via the Claude API. Evaluate both against the Phase 1 manifest and
report precision/recall/F1.
```

**Verify**
```bash
python eval/run_eval.py --stage detection
```

---

### Phase 4 — Provenance-gated retrieval policy engine *(use Plan Mode)*

**Claude Code prompt**
```
Implement Phase 4: src/retrieval/policy_engine.py that gates retrieved
candidates on trust_tier and poison_score, flags retrieval-frequency
anomalies, and exposes retrieve(query) as a LangGraph node. Every threshold
must be config-driven, not hardcoded.
```

**Verify**
```bash
python eval/run_eval.py --stage retrieval --compare-baseline
```

**Exit criteria:** ≥30-point ASR reduction at <15% retrieval-precision loss vs. undefended baseline. **This is the minor-project gate.**

---

### Phase 5 — Generation, faithfulness check, minor-project eval

**Claude Code prompt**
```
Implement Phase 5: src/generation/generate.py as a LangGraph node calling
the configured LLM provider (OpenAI by default, swappable via LLM_PROVIDER
— see src/generation/llm_client.py) with gated context,
src/generation/faithfulness.py doing an NLI check per generated claim,
eval/run_eval.py producing the full ASR / precision / F1 / latency report,
and a FastAPI /query endpoint in src/api/main.py wiring the whole loop
end-to-end.
```

**Verify**
```bash
uvicorn src.api.main:app --reload &
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "test question"}'
python eval/run_eval.py --full-report
```

**Checkpoint**
```bash
git add -A && git commit -m "feat: complete minor-project RAG loop"
git tag v0.1-minor
```

> **Non-code bridge:** file the Indian provisional patent on provenance-gated retrieval *before* publishing these eval results anywhere public. Keep the repo private until it's filed.

---

### Phase 6 — Verifiable unlearning + deletion receipts *(use Plan Mode)*

**Claude Code prompt**
```
Implement Phase 6: src/unlearning/delete.py for delete-by-ID against
Qdrant with index compaction, src/unlearning/receipt.py issuing a signed
JSON deletion receipt with the same Ed25519 setup as Phase 2, and
src/unlearning/ghost_vector_defense.py defending against post-delete
embedding reconstruction. Write an eval script that attempts reconstruction
via nearest-neighbor search and confirms failure above a chance baseline.
Benchmark deletion latency.
```

**Verify**
```bash
python -m src.unlearning.delete --chunk-id <id>
python eval/run_eval.py --stage unlearning
```

**Checkpoint**
```bash
git tag v0.2-unlearning
```

---

### Phase 7 — LLMOps dashboard & observability

**Claude Code prompt**
```
Implement Phase 7: OpenTelemetry spans across ingestion, retrieval, and
generation (src/observability/otel_setup.py), and a Streamlit dashboard in
dashboard/ showing live query traces, detection rate over time, trust-tier
distribution, and the deletion audit log.
```

**Verify**
```bash
streamlit run dashboard/app.py
```

---

### Phase 8 — Agentic-RAG coverage, multi-tenancy, DPDP mapping *(use Plan Mode)*

**Claude Code prompt**
```
Implement Phase 8: a small multi-agent LangGraph (planner -> parallel
retriever agents -> synthesizer), re-run the Phase 4/5 poisoning eval
against it and report ASR/F1 deltas, add tenant_id to the vector schema
with an isolation test, and draft docs/dpdp_compliance_mapping.md mapping
dashboard/audit fields to DPDP Rules 2025 obligations.
```

**Verify**
```bash
python eval/run_eval.py --stage agentic
pytest tests/ -k tenant_isolation
```

---

### Phase 9 — Packaging, CI/CD, open-core split, demo polish

**Claude Code prompt**
```
Implement Phase 9: production Dockerfiles, a full-stack
docker-compose.prod.yml, a GitHub Actions CI workflow (lint, pytest, eval
smoke-test), and a script that toggles each defensive layer on/off
individually to produce a cost-accuracy frontier report.
```

**Checkpoint**
```bash
git tag v1.0-major
```

Move to Section 4 for the actual deployment steps.

---

## 4. Deployment

### 4.1 Run the full stack locally

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### 4.2 Example production Dockerfile (API service)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .
COPY src ./src
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.3 Example `docker-compose.prod.yml` shape

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    volumes: ["qdrant_data:/qdrant/storage"]
  api:
    build: ./
    env_file: .env
    depends_on: [qdrant]
    ports: ["8000:8000"]
  dashboard:
    build: ./dashboard
    depends_on: [api]
    ports: ["8501:8501"]
volumes:
  qdrant_data:
```

### 4.4 CI — GitHub Actions (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest
      - run: python eval/run_eval.py --smoke
```

### 4.5 Live demo during a review (no hosting needed)

Run the stack locally, then expose it temporarily:
```bash
# Option A: cloudflared
cloudflared tunnel --url http://localhost:8000

# Option B: ngrok
ngrok http 8000
```
Either gives you a public HTTPS URL for the duration of the tunnel — enough for a live viva demo without deploying anywhere.

### 4.6 Longer-term hosting

For a persistent demo beyond a single review, a small VM (college server, or a free/low-cost tier on a cloud provider) running `docker compose -f docker-compose.prod.yml up -d` behind a reverse proxy (Caddy handles HTTPS automatically) is the simplest path:
```bash
sudo apt install caddy
# Caddyfile: yourdomain.com { reverse_proxy localhost:8000 }
sudo systemctl restart caddy
```
Platforms like Railway, Render, or Fly.io can also run this compose stack directly — check their current free-tier limits before committing, since those change often.

---

## 5. Command cheat sheet

| Task | Command |
|---|---|
| Start Claude Code | `claude` |
| Check Claude Code health | `claude doctor` |
| Start Qdrant only | `docker compose up -d qdrant` |
| Run all tests | `pytest` |
| Lint | `ruff check .` |
| Run eval suite | `python eval/run_eval.py --full-report` |
| Run API locally | `uvicorn src.api.main:app --reload` |
| Run dashboard | `streamlit run dashboard/app.py` |
| Full stack up | `docker compose -f docker-compose.prod.yml up -d --build` |
| Tunnel for a demo | `cloudflared tunnel --url http://localhost:8000` |
| Tag a milestone | `git tag vX.Y-name && git push --tags` |

---

## 6. Claude Code prompt library (all in one place)

Phase 0: *"Set up the repository structure for VeritasRAG... [Section 3, Phase 0]"*
Phase 1: *"Implement Phase 1: src/ingestion/loaders.py... [Section 3, Phase 1]"*
Phase 2: *"Implement Phase 2: src/embeddings/embedder.py... [Section 3, Phase 2]"*
Phase 3: *"Implement Phase 3: perplexity scoring... [Section 3, Phase 3]"*
Phase 4: *"Implement Phase 4: src/retrieval/policy_engine.py... [Section 3, Phase 4]"*
Phase 5: *"Implement Phase 5: src/generation/generate.py... [Section 3, Phase 5]"*
Phase 6: *"Implement Phase 6: src/unlearning/delete.py... [Section 3, Phase 6]"*
Phase 7: *"Implement Phase 7: OpenTelemetry spans... [Section 3, Phase 7]"*
Phase 8: *"Implement Phase 8: a small multi-agent LangGraph... [Section 3, Phase 8]"*
Phase 9: *"Implement Phase 9: production Dockerfiles... [Section 3, Phase 9]"*

Session-end prompt, every phase: *"Update PROGRESS.md — what's done, what's outstanding, today's eval numbers."*

End-of-session commit prompt: *"Write a commit message from the actual diff and commit it."*
