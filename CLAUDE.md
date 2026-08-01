# VeritasRAG — project memory

This file is auto-loaded by Claude Code at the start of every session in this repo. Keep it short and factual; detailed phase instructions live in `PLAN.md`.

## What this project is

VeritasRAG is a trustworthy RAG integrity layer combining knowledge-base poisoning detection, cryptographic document provenance / trust-tier gating, and verifiable vector-database unlearning. Full context, tech stack, architecture, and the phase-by-phase build plan are in `PLAN.md` — read it before starting work if you haven't already this session.

## Current phase

Check `PROGRESS.md` for what's done and what's outstanding. Don't start a new phase from `PLAN.md` unless the user explicitly says so — confirm which phase we're on if it isn't obvious from `PROGRESS.md`.

## Conventions

- Python 3.11+, type hints on all function signatures, docstrings on public functions.
- Formatting/linting: `ruff` for both. Run it before considering a task done.
- Tests go in `tests/`, mirroring the `src/` layout. Every new module gets at least a sanity test.
- All randomness in `eval/` and synthetic-data generation must be seeded — reproducibility matters more than elegance here.
- **Never hand-roll cryptography.** Signing, hashing, and verification go through the `cryptography` package only. Flag it and ask before implementing any crypto-adjacent logic a different way.
- Don't hardcode thresholds (trust scores, ASR targets, detection cutoffs) — read them from config so they can be swept for the cost-accuracy frontier report in Phase 9.
- No secrets, API keys, or credentials in code or commits. Use `.env` (already gitignored) and `python-dotenv`.

## Environment notes

- WSL2's `/tmp` is a small tmpfs — `TMPDIR` is set to `~/tmp` in `.bashrc` to avoid "No space left on device" during large pip installs.

## Working style for this project

- Work within the scope of the current phase in `PLAN.md`. If something outside that phase's scope seems necessary, flag it rather than silently expanding scope.
- For phases marked "Use Plan Mode" in `PLAN.md`, propose an approach and wait for approval before editing files.
- At the end of a session, update `PROGRESS.md`: what was completed, what's outstanding, and any eval numbers produced that session.
- When running the eval harness, report the actual numbers in your response — don't just say "eval passed."

## Repo layout

See `PLAN.md` Part 3 for the full intended structure. If you need to deviate from it, say so explicitly rather than silently reorganizing.
