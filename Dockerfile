# Production images for VeritasRAG's two long-running services: the FastAPI
# query API (target: api, PLAN.md Section 4.2/4.4) and the Streamlit
# dashboard (target: dashboard). One Dockerfile, two final targets sharing a
# common `deps`/`runtime-base` stage -- both services transitively import
# nearly the entire dependency set already (dashboard/data.py ->
# src.vectorstore.qdrant_client -> src.embeddings.embedder ->
# sentence-transformers, even though the dashboard itself never calls
# embed()), so splitting the pip-install logic across two separate
# Dockerfiles would just duplicate it for no real image-size win.
# docker-compose.prod.yml builds each service with `target: api` /
# `target: dashboard` and build context `.` (the repo root -- the dashboard
# needs root-level `src/`, so it can't use PLAN.md's literal
# `build: ./dashboard` shape; flagged here per CLAUDE.md's "say so
# explicitly" rule for deviations from PLAN.md).
#
# The local `veritasrag` package is never pip-installed into the image --
# PYTHONPATH=/app is set instead, the same import convention
# `pyproject.toml`'s `[tool.pytest.ini_options] pythonpath = ["."]` already
# uses for every test/eval script in this repo (see PROGRESS.md's Phase 7
# packaging note). `pyproject.toml` does now also support a real
# `pip install .` (Phase 9 fix: `[tool.hatch.build.targets.wheel] packages`
# was missing), but PYTHONPATH keeps this image's behavior identical to how
# every phase has already run and tested this code -- one less thing that
# could behave differently in the container than on a dev machine.

FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are installed straight from pyproject.toml's
# [project.dependencies] list (extracted with stdlib tomllib -- no separate
# requirements.txt to drift out of sync with the single source of truth)
# *before* the source tree is copied in, so a source-only code change never
# invalidates this layer. This is the expensive step (torch/transformers/
# faiss-cpu), so keeping it cached across rebuilds matters.
COPY pyproject.toml ./
RUN pip install --upgrade pip \
    # torch is only ever used here for local CPU inference (distilgpt2 /
    # bge-small-en / nli-deberta-v3-xsmall -- every model choice in this repo
    # is picked for a "CPU-only box", per CLAUDE.md/PROGRESS.md). Installing
    # it from PyPI's default index pulls the full CUDA/cuDNN/triton GPU stack
    # (~2GB of libraries this project never uses). Installing the CPU-only
    # wheel first means the transitive install below (via
    # sentence-transformers/transformers) finds torch already satisfied and
    # never replaces it with the GPU build.
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -c "\
import tomllib; \
deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
print(chr(10).join(deps))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt


FROM deps AS runtime-base

# curl only needed for the HEALTHCHECKs below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

COPY src ./src
COPY dashboard ./dashboard
COPY eval ./eval
COPY pyproject.toml ./

# Generated-artifact directories (keys/, models/, receipts/, observability/,
# eval/benchmarks/) are gitignored and created on demand in local dev --
# pre-create + chown them here so the non-root user below can write to them
# even before a volume is mounted over them at runtime.
RUN useradd --create-home --uid 1000 veritasrag \
    && mkdir -p keys models receipts observability eval/benchmarks .cache/huggingface \
    && chown -R veritasrag:veritasrag /app
USER veritasrag


FROM runtime-base AS api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM runtime-base AS dashboard

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true"]
