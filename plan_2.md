# VeritasRAG — Implementation Handbook

A phase-by-phase build guide for the VeritasRAG Trustworthy RAG Integrity Layer, written to be used directly with **Claude Code in VS Code**. Drop this file in your repo root as `plan_2.md`. Pair it with `CLAUDE.md` (provided separately) so Claude Code auto-loads your conventions every session.

---

## Part 0 — How to work through this handbook with Claude Code

**One phase per working session.** Each phase below is scoped to roughly one sitting. Don't hand Claude Code the whole handbook and say "build it" — that produces a shallow pass over everything. Instead, each session:

1. Open the project folder in VS Code, open the Claude Code panel (Spark icon in the editor toolbar, or `Cmd/Ctrl+Shift+P` → "Claude Code: Open in New Tab").
2. Tell it explicitly which phase you're on: *"Read plan_2.md and CLAUDE.md. We're starting Phase 3. Only work within that phase's scope — don't jump ahead."*
3. For phases with real design decisions (3, 4, 6, 8 below), switch to **Plan Mode** first (mode indicator at the bottom of the prompt box) so Claude Code proposes an approach before touching files. Read the plan, correct anything off, then approve.
4. Let it work, review the diffs it proposes (VS Code shows a side-by-side diff before every edit), and course-correct early rather than letting it run far off track.
5. At the end of the session: *"Update PROGRESS.md — what's done, what's outstanding, today's eval numbers if any."* This is what keeps continuity across sessions once context gets compacted or you start a fresh session tomorrow.
6. Commit. Ask Claude Code to write the commit message from the actual diff rather than writing one yourself.

**Why phase-scoped sessions matter:** long single-session builds degrade — context fills up, earlier decisions get compacted away, and Claude Code starts re-deriving things it already decided. Fresh, well-scoped sessions with `plan_2.md` + `CLAUDE.md` + `PROGRESS.md` as anchors consistently produce better code than one marathon session.

**Keep `eval/` deterministic.** Seed every random generator used in the synthetic poison corpus and the eval harness. This means every phase's numbers are reproducible and you can tell at a glance whether a Claude Code change helped or hurt.

---

## Part 1 — Complete tech stack

| Layer | Component | Tool / library | Notes |
|---|---|---|---|
| Language / runtime | Core backend | Python 3.11+ | |
| | Dashboard (optional later) | Node.js 22+ | Only if you migrate the dashboard to Next.js in Phase 7 |
| Orchestration | Agent graph | LangGraph (LangChain 1.x, `langchain.agents.create_agent`) | Matches your existing skillset |
| Vector database | Storage + retrieval | Qdrant | Run locally via Docker; Milvus is a documented fallback if you need it |
| Embeddings | Chunk + query embedding | HuggingFace `sentence-transformers` (e.g. BGE or E5 family) | Local, no per-call API cost — important for a student budget |
| Generation | Answer synthesis, LLM-as-judge baseline | Claude via the Anthropic API | Use a small, cheap model for the perplexity/judge baselines and a stronger one only for final generation, to control API spend |
| Backend API | Service layer | FastAPI | |
| Poison detection | Perplexity scoring | Small HF causal LM (e.g. GPT-2-class) | Cheap to run locally, good anomaly signal |
| | Cluster anomaly detection | scikit-learn / faiss | Outlier detection in embedding space |
| | Optional classifier | HF `transformers` `Trainer` | Fine-tuned on your Phase 1 synthetic-poison labels |
| Faithfulness / NLI | Citation-faithfulness check | HF cross-encoder NLI model | Flags generated claims not entailed by retrieved chunks |
| Provenance & crypto | Signing, deletion receipts | Python `cryptography` (Ed25519) | Never hand-roll signature schemes — see `CLAUDE.md` |
| Evaluation | ASR, F1, latency harness | Ragas and/or Arize Phoenix + custom scripts | |
| Observability | Tracing | OpenTelemetry (Python SDK) | Export to console locally, Jaeger/Phoenix if you want a UI |
| Dashboard | LLMOps UI | Streamlit first, Next.js later if time allows | Streamlit gets you a working dashboard in hours, not days |
| Containerization | Local dev + deployment | Docker + Docker Compose, WSL2 | Matches your existing environment |
| CI/CD | Automated checks | GitHub Actions | Lint, pytest, eval smoke-test on every PR |
| Testing | Unit/integration | pytest | |
| Datasets | Poison benchmark source | Natural Questions, HotpotQA (via HF `datasets`) | |
| | Unlearning eval | TOFU-style suite | For membership-inference-style checks in Phase 6 |

---

## Part 2 — Architecture: how everything connects

The diagram above shows the shape of it. In words:

A document enters through the **ingestion gateway**, which loads it, strips PII, runs the poison pre-scan, and signs its content hash for provenance before it's ever embedded. Chunks land in **Qdrant** carrying trust-tier metadata (`verified` / `unverified` / `quarantined`) alongside the vector itself — trust isn't a side table, it's on the same payload the retriever reads.

At query time, the **retrieval policy engine** sits between similarity search and the LLM: it down-weights or drops low-trust chunks and watches for retrieval-frequency anomalies (a chunk getting pulled far more often than its content justifies is a poisoning tell). Only what survives that gate reaches **generation**, where the LLM answers and a citation-faithfulness checker confirms every claim is actually entailed by what was retrieved.

Two services sit off the critical path but are wired into the same store: the **unlearning service** deletes-by-ID directly against Qdrant and issues a signed deletion receipt, and **observability** taps the gateway and policy engine to log traces, detection rates, and the deletion audit trail that the dashboard renders.

---

## Part 3 — Repository structure

Ask Claude Code to scaffold exactly this in Phase 0:

```
veritasrag/
├── CLAUDE.md
├── PLAN.md                  # this handbook
├── PROGRESS.md              # running log, updated every session
├── pyproject.toml
├── docker-compose.yml
├── .github/workflows/ci.yml
├── src/
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── redaction.py
│   │   ├── poison_scan.py
│   │   └── provenance.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── vectorstore/
│   │   ├── qdrant_client.py
│   │   └── schema.py
│   ├── detection/
│   │   ├── perplexity.py
│   │   ├── cluster_anomaly.py
│   │   └── classifier.py
│   ├── retrieval/
│   │   ├── policy_engine.py
│   │   └── trust_scoring.py
│   ├── generation/
│   │   ├── generate.py
│   │   └── faithfulness.py
│   ├── unlearning/
│   │   ├── delete.py
│   │   ├── receipt.py
│   │   └── ghost_vector_defense.py
│   ├── observability/
│   │   └── otel_setup.py
│   └── api/
│       └── main.py
├── dashboard/
├── eval/
│   ├── synthetic_poison_generator.py
│   ├── benchmarks/
│   └── run_eval.py
├── tests/
└── docs/
```

---

## Part 4 — Phase-by-phase plan

Each phase lists what it builds, what it needs from earlier phases, a ready-to-paste Claude Code prompt, and the exit criteria that tell you it's actually done — not just "code exists."

### Phase 0 — Environment & repo scaffold
**Time:** 0.5–1 day · **Builds on:** nothing

**Tasks**
- Install Claude Code, Docker/WSL2, Python 3.11+.
- Scaffold the repo structure in Part 3.
- `docker-compose.yml` running Qdrant locally.
- `pyproject.toml` / `requirements.txt` with the Part 1 stack.
- Drop in `CLAUDE.md` and this `PLAN.md`; create an empty `PROGRESS.md`.

**Claude Code prompt**
```
Set up the initial repository for VeritasRAG as described in plan_2.md Part 3.
Create the folder layout, a pyproject.toml with the dependencies from the
Tech Stack table in plan_2.md Part 1, a docker-compose.yml that runs Qdrant
locally, and a .gitignore for a Python project. Scaffolding only — no
business logic yet. After scaffolding, run docker compose up -d and verify
Qdrant responds.
```

**Exit criteria:** `docker compose up -d` starts Qdrant; `curl localhost:6333/collections` returns 200; repo initialized and pushed (private) to GitHub.

---

### Phase 1 — Data ingestion + synthetic poisoned corpus
**Time:** 2–3 days · **Builds on:** Phase 0

This is the single most important early deliverable — every later eval depends on this benchmark existing and being reproducible.

**Tasks**
- Document loaders for text/PDF/HTML (LangChain loaders) + chunking.
- `eval/synthetic_poison_generator.py`: pull a subset of Natural Questions / HotpotQA via HF `datasets`, generate PoisonedRAG-style adversarial passages, inject N per target question, and save a manifest with ground-truth poison labels.

**Claude Code prompt**
```
Implement Phase 1 from plan_2.md: src/ingestion/loaders.py for text/PDF/HTML
using LangChain document loaders, and eval/synthetic_poison_generator.py
that pulls a subset of Natural Questions from Hugging Face datasets,
generates PoisonedRAG-style adversarial passages (crafted text designed to
be retrieved for a target question and steer the answer), injects them into
the corpus, and saves a JSON manifest of
{chunk_id, text, is_poisoned, target_question, source_doc}. Seed all
randomness. Write pytest tests for the loader and a sanity check that
manifest labels match the injected chunks.
```

**Exit criteria:** pytest green; manifest has ≥200 clean + ≥50 poisoned chunks; a second run with the same seed produces an identical manifest.

---

### Phase 2 — Embeddings & vector DB with trust-tier schema
**Time:** 2 days · **Builds on:** Phase 1

**Tasks**
- `embeddings/embedder.py` wrapping a sentence-transformer model.
- Qdrant collection schema: `doc_id`, `chunk_id`, `source_uri`, `trust_tier` (verified/unverified/quarantined), `provenance_signature`, `ingested_at`, `content_hash`.
- Batch upsert pipeline from Phase 1 output.
- `ingestion/provenance.py`: sign each chunk's content hash (Ed25519 via `cryptography`), store the signature and a public-key reference in the payload.

**Claude Code prompt**
```
Implement Phase 2 from plan_2.md: src/embeddings/embedder.py wrapping a
sentence-transformers model, src/vectorstore/schema.py and
qdrant_client.py defining and upserting the collection schema described in
plan_2.md Phase 2, and src/ingestion/provenance.py that signs each chunk's
content hash with Ed25519 using the `cryptography` package (do not
hand-roll any crypto). Batch-upsert the Phase 1 corpus. Write a script that
samples a chunk from Qdrant and verifies its stored signature against its
content hash.
```

**Exit criteria:** full corpus upserted with all metadata fields populated; signature-verification script passes on a random sample.

---

### Phase 3 — Poison detection module
**Time:** 3–4 days · **Builds on:** Phase 2 · **Use Plan Mode**

**Tasks**
- `detection/perplexity.py`: perplexity scoring with a small causal LM.
- `detection/cluster_anomaly.py`: embedding-space outlier detection.
- `detection/classifier.py` (optional): lightweight classifier fine-tuned on Phase 1 labels.
- LLM-as-judge baseline via the Claude API for comparison.
- `detection/pipeline.py` combining signals into a per-chunk poison score.

**Claude Code prompt**
```
Implement Phase 3 from plan_2.md. Build the poison detector combining
perplexity scoring (src/detection/perplexity.py), embedding-space cluster
anomaly detection (src/detection/cluster_anomaly.py), and an optional
fine-tuned classifier (src/detection/classifier.py), fused into a single
poison_score per chunk in src/detection/pipeline.py. Also implement an
LLM-as-judge baseline that scores the same chunks via the Claude API for
comparison. Evaluate all of it against the Phase 1 manifest and report
precision, recall, and F1 for each method.
```

**Exit criteria:** F1/precision/recall report comparing your combined detector against the LLM-as-judge baseline on the Phase 1 corpus.

---

### Phase 4 — Provenance-gated retrieval policy engine
**Time:** 2–3 days · **Builds on:** Phase 3 · **Use Plan Mode**

This is the module the patent's first independent claim rests on — gate carefully and document the design decisions.

**Tasks**
- `retrieval/policy_engine.py`: down-weight or quarantine chunks below a trust threshold or flagged by the Phase 3 detector.
- Retrieval-frequency anomaly detection (a chunk retrieved unusually often across unrelated queries is a poisoning tell).
- A `retrieve()` function pluggable into a LangGraph node.

**Claude Code prompt**
```
Implement Phase 4 from plan_2.md: src/retrieval/policy_engine.py that takes
retrieved candidates from Qdrant, applies trust-tier gating using the
poison_score from Phase 3 and the trust_tier metadata from Phase 2, and
flags chunks with anomalous retrieval frequency across unrelated queries.
Expose a retrieve(query) function usable as a LangGraph node. Make every
threshold configurable, not hardcoded, so we can later measure a
cost-accuracy frontier.
```

**Exit criteria:** with gating enabled, measure attack success rate (ASR) reduction and retrieval-precision loss on the Phase 1 corpus versus an undefended baseline. **Target: ≥30-point ASR reduction at <15% retrieval-precision loss** — this is the minor-project gate.

---

### Phase 5 — Generation, citation-faithfulness check, minor-project eval
**Time:** 2–3 days · **Builds on:** Phase 4

**Tasks**
- `generation/generate.py`: LangGraph node calling Claude for the final answer using gated context.
- `generation/faithfulness.py`: NLI check that generated claims are entailed by cited chunks.
- `eval/run_eval.py`: full pipeline eval producing the ASR / precision / F1 / latency table for the paper.
- `api/main.py`: FastAPI `/query` endpoint wiring the whole loop together.

**Claude Code prompt**
```
Implement Phase 5 from plan_2.md: src/generation/generate.py as a LangGraph
node that generates an answer from Claude using the gated retrieval
context, src/generation/faithfulness.py that NLI-checks each generated
claim against its cited chunk, eval/run_eval.py producing a full report
(ASR, precision, F1, latency) across the whole pipeline, and a FastAPI
/query endpoint in src/api/main.py that returns an answer with citations
and trust scores end-to-end.
```

**Exit criteria:** `/query` returns a full answer with citations and trust scores; `eval/run_eval.py` produces the report table. **Tag the repo `v0.1-minor` — this is the minor-project deliverable.**

> **Bridge point (not a coding phase):** file the Indian provisional patent on provenance-gated retrieval *before* publishing the Phase 5 eval results anywhere public — before a paper submission, before a public GitHub release, before a conference poster. Keep the repo private until it's filed. See the earlier roadmap for the reasoning.

---

### Phase 6 — Verifiable unlearning + deletion receipts
**Time:** 3–4 days · **Builds on:** Phase 5 · **Use Plan Mode**

This opens the major project and is the strongest remaining patentable claim.

**Tasks**
- `unlearning/delete.py`: hard-delete from Qdrant + trigger index compaction/rebuild.
- `unlearning/receipt.py`: signed JSON deletion receipt (`chunk_id`, `deleted_at`, `requester`, `signature`).
- `unlearning/ghost_vector_defense.py`: defend against post-delete embedding reconstruction (periodic index rebuild, or a membership-inference-style check that confirms deleted content can't be recovered from nearest neighbors).

**Claude Code prompt**
```
Implement Phase 6 from plan_2.md: src/unlearning/delete.py for delete-by-ID
against Qdrant with index compaction, src/unlearning/receipt.py that
issues a signed JSON deletion receipt using the same Ed25519 setup as
Phase 2's provenance signer, and src/unlearning/ghost_vector_defense.py
that defends against embedding-reconstruction after deletion. Write an eval
script that attempts to reconstruct a deleted chunk via nearest-neighbor
search and confirms it fails above a chance baseline. Benchmark deletion
latency.
```

**Exit criteria:** deletion receipt issued and independently verifiable; reconstruction-attempt eval shows deleted content isn't recoverable; deletion latency benchmarked.

---

### Phase 7 — LLMOps dashboard & observability
**Time:** 3–4 days · **Builds on:** Phase 6

**Tasks**
- `observability/otel_setup.py`: OpenTelemetry spans across ingestion, retrieval, generation.
- Wire Ragas/Phoenix for ongoing eval-in-production.
- `dashboard/`: Streamlit app showing live query traces, detection rate over time, trust-tier distribution, deletion audit log.

**Claude Code prompt**
```
Implement Phase 7 from plan_2.md: instrument the ingestion, retrieval, and
generation modules with OpenTelemetry spans (src/observability/otel_setup.py),
and build a Streamlit dashboard in dashboard/ that shows live query traces,
poison-detection rate over time, trust-tier distribution across the
corpus, and the deletion audit log from Phase 6.
```

**Exit criteria:** dashboard reflects at least one full query cycle and one deletion event with real data, not mocked.

---

### Phase 8 — Agentic-RAG coverage, multi-tenancy, DPDP mapping
**Time:** 4–5 days · **Builds on:** Phase 7 · **Use Plan Mode**

**Tasks**
- A small agentic RAG graph (planner → multiple retriever agents → synthesizer) and re-run the Phase 4/5 poisoning eval against it — this directly targets the literature's stated gap.
- `tenant_id` on the schema + isolation tests.
- A DPDP Rules 2025 compliance-mapping document: which dashboard fields satisfy which obligations (data-principal requests, deletion audit trail, breach logging).

**Claude Code prompt**
```
Implement Phase 8 from plan_2.md: build a small multi-agent LangGraph
(planner -> parallel retriever agents -> synthesizer), re-run the
poisoning eval from Phase 4/5 against this agentic architecture and report
whether ASR or detection F1 change, add tenant_id to the vector store
schema with an isolation test, and draft docs/dpdp_compliance_mapping.md
mapping each dashboard/audit field to the DPDP Rules 2025 obligation it
satisfies.
```

**Exit criteria:** agentic-RAG poisoning eval report exists; tenant isolation test passes; DPDP mapping doc complete.

---

### Phase 9 — Packaging, CI/CD, open-core split, demo polish
**Time:** 3–4 days · **Builds on:** Phase 8

**Tasks**
- Production Dockerfile(s) + full-stack `docker-compose.yml`.
- GitHub Actions CI: lint, pytest, an eval smoke-test on every PR.
- If pursuing open-core licensing: split the repo (or use feature flags) so the detector + provenance gate are open, unlearning + dashboard + DPDP mapping are gated.
- Final cost-accuracy frontier report — every layer from Phase 4 onward should already be toggleable, so this is a measurement pass, not new code.
- Demo script.

**Claude Code prompt**
```
Implement Phase 9 from plan_2.md: production Dockerfiles and a full-stack
docker-compose.yml, a GitHub Actions workflow that runs lint, pytest, and
an eval smoke-test on every PR, and a script that runs the full pipeline
with each defensive layer toggled on/off individually to produce a
cost-accuracy frontier report.
```

**Exit criteria:** `docker compose up` brings up the full stack from a clean clone; CI is green; cost-accuracy frontier report generated; demo rehearsed.

---

## Part 5 — Claude Code operating notes specific to this project

- **Plan Mode for phases 3, 4, 6, 8** — these involve real design tradeoffs, not just plumbing. Review the plan before approving.
- **Never let it hand-roll crypto.** Every signing operation should go through the `cryptography` package. This is already stated in `CLAUDE.md`, but repeat it if you see anything suspicious in a diff.
- **Keep `eval/` deterministic and seeded** so every phase's numbers are directly comparable across sessions.
- **Update `PROGRESS.md` at the end of every session** — this is what carries context forward once a session ends or compacts.
- **Keep the repo private until the provisional patent is filed** (after Phase 5, before any public paper submission or release).

---

## Part 6 — Phase-to-roadmap mapping

| Phases | Maps to |
|---|---|
| 0–5 | Minor project (prototype) |
| Bridge after 5 | Paper submission + provisional patent filing |
| 6–9 | Major project (production) |
| After 9 | Validate & position, then launch & fundraising — see the earlier roadmap |
