# DPDP Rules 2025 — technical control mapping

## Scope and disclaimer

This document maps VeritasRAG's existing technical audit surface — the
fields and modules already built in Phases 2, 6, and 7 — to obligation
*categories* under India's Digital Personal Data Protection Act, 2023 and
its Rules (referred to here as "DPDP Rules 2025"). It is engineering
documentation for internal/product reference, written by an engineer, not a
lawyer.

**It is not a legal compliance certification.** Obligation categories below
are described in plain language, deliberately without citing specific
section/rule numbers, since the exact numbering and final text of the Rules
should be verified against the current gazetted version before this
document is used for any external, regulatory, or contractual purpose. A
real deployment claiming DPDP compliance needs legal review of this mapping,
not just the engineering self-assessment below.

The mapping is grounded only in fields and modules that actually exist in
this codebase today — no coverage is claimed that isn't backed by real,
inspectable code.

## Mapping table

| DPDP obligation category | VeritasRAG technical control | Field / module | Status |
|---|---|---|---|
| Right to erasure | Delete-by-ID against Qdrant with forced index compaction, so a deletion is not just a soft tombstone | `src/unlearning/delete.py::delete_with_receipt()` | Implemented |
| Verifiable proof of erasure | Signed (Ed25519) deletion receipt recording what was deleted and what was verified about the deletion, independently re-verifiable at any time | `src/unlearning/receipt.py::RECEIPT_FIELDS` (`chunk_id`, `point_id`, `content_hash`, `found`, `deleted_at`, `requester`, `membership_check_passed`, `reconstruction_check_passed`, `receipt_hash`, `signature`), `verify_receipt()` | Implemented |
| Erasure audit trail, reviewable on demand | Every receipt persisted to disk and surfaced in a queryable audit log, with validity **recomputed live** rather than trusted from a stored field | `src/unlearning/receipt.py::save_receipt()` / `load_receipt()`, `dashboard/data.py::verified_receipts()` | Implemented |
| Protection against reconstruction of "erased" data | Post-deletion checks that the deleted vector cannot be summoned back via its own embedding (membership-inference self-check) and that no surviving chunk leaks equivalent content beyond a chance baseline | `src/unlearning/ghost_vector_defense.py::verify_unrecoverable()` / `attempt_reconstruction()` | Implemented |
| Reasonable security safeguards — data integrity / provenance | Every stored chunk is content-hashed and Ed25519-signed at ingestion; tampering with text or hash is independently detectable | `src/ingestion/provenance.py::ProvenanceSigner`, `src/vectorstore/qdrant_client.py::verify_sample()` | Implemented |
| Reasonable security safeguards — data segregation | Per-tenant filtering on every vector-store read, so one tenant's data cannot be retrieved under another tenant's query | `src/vectorstore/schema.py::DEFAULT_TENANT_ID`, `src/vectorstore/qdrant_client.py::search(tenant_id=...)`, `src/retrieval/policy_engine.py::retrieve(tenant_id=...)` — see the caveat below | Implemented (server-side filter only; see "Known gap" below) |
| Reasonable security safeguards — content-integrity gating | Trust-tier and poison-score gating on every retrieval, quarantining or dropping untrustworthy/poisoned content before it reaches generation | `src/retrieval/policy_engine.py::retrieve()`/`_gate()` | Implemented (this project's core defense layer, not itself a DPDP-mandated control, but relevant to "reasonable safeguards against unauthorized processing/corruption") |
| Auditability / accountability — processing-decision logging | Every retrieval call emits a structured trace recording how many candidates were fetched, allowed, downweighted, or dropped, and why | OTel spans `veritasrag.retrieval.retrieve` — attributes `veritasrag.retrieval.{fetched,allow,downweight,drop}_count`, `veritasrag.retrieval.drop_reason.*_count` (`src/retrieval/policy_engine.py`) | Implemented |
| Auditability / accountability — live data-inventory visibility | On-demand snapshot of how much data sits in each trust tier | `src/vectorstore/qdrant_client.py::trust_tier_counts()`, `dashboard/data.py::live_trust_tier_distribution()` | Implemented |
| Auditability / accountability — end-to-end request tracing | Every query traced end to end (retrieval → generation → faithfulness check) with parent/child span linkage, viewable per-request | `src/observability/otel_setup.py`, `dashboard/app.py`'s "Live Query Traces" panel | Implemented |

## Known gap, flagged rather than hidden

**Erasure/segregation interaction**: `src/vectorstore/qdrant_client.py::point_id_for_chunk()` derives a stored point's Qdrant ID from `chunk_id` alone (`uuid5(NAMESPACE_URL, chunk_id)`), not from `(tenant_id, chunk_id)`. Two tenants that happened to share a `chunk_id` would silently overwrite each other's point on upsert, and — more relevant to this document — a delete-by-ID erasure request is not itself tenant-scoped: `src/unlearning/delete.py` operates on a bare `chunk_id`/`point_id` with no check that the requester's tenant actually owns that point. In a real multi-tenant deployment, an erasure request should be verified against the requesting tenant before it is honored. This is a deliberate, documented Phase 8 limitation (see `PROGRESS.md`), not an oversight — fixing the point-ID scheme would change every existing point ID in the live corpus, a larger change than this phase's literal scope of adding `tenant_id` to the schema.

**Tenant identity is not authenticated anywhere in this codebase.** `tenant_id` is a parameter callers pass to `retrieve()`/`search()`; there is no auth/identity layer establishing which tenant a caller actually is. The FastAPI `/query` endpoint (`src/api/main.py`) does not accept or enforce a `tenant_id` at all — this was a deliberate Phase 8 scope decision (see `PROGRESS.md`): a client-self-reported tenant field on an unauthenticated endpoint would be a *false* segregation control, not a partial one. Any real deployment relying on this mapping's "data segregation" row needs a genuine identity layer in front of retrieval before that row can be considered actually enforced end-to-end, not just structurally supported.

## Gaps — not applicable to this deployment shape

The following obligation categories have no corresponding VeritasRAG control today. They are not silently assumed covered by anything above.

| DPDP obligation category | Status | Reason |
|---|---|---|
| Consent capture / purpose limitation at collection time | Gap | VeritasRAG ingests a synthetic/research corpus (Phase 1); there is no end-user-facing data-collection surface, so no consent-capture UI or purpose-binding record exists. |
| Data Principal rights — access / correction requests | Gap | Only erasure is implemented (Phase 6). There is no "show me everything you hold about me" or correction workflow — this project's data model (document chunks, not per-person profiles) doesn't map cleanly onto per-Data-Principal records without a separate identity-to-chunk linkage this project does not build. |
| Breach detection and notification pipeline | Gap | No breach-detection tooling or notification workflow exists. The OTel traces (Phase 7) could feed a future anomaly-detection/alerting layer, but no such layer has been built. |
| Grievance officer / grievance redressal workflow | Gap | No user-facing grievance-intake mechanism exists in this project. |
| Consent Manager integration | Gap | Not applicable — this project has no consent-collection surface to integrate a Consent Manager against. |
| Significant Data Fiduciary obligations (DPO appointment, independent data audit, periodic DPIA) | Gap | Organizational/process obligations, not a code-level control; out of an engineering repo's scope regardless of deployment scale. |
| Cross-border data transfer restrictions | Not applicable to current deployment | This project runs a single local Qdrant instance with no cross-border data movement; the control would only become relevant if deployed against a hosted, geographically-distributed backend. |
| Children's-data safeguards (verifiable parental consent, no behavioral tracking/targeted ads directed at children) | Not applicable to current deployment | VeritasRAG has no end-user-facing product surface and does not knowingly process children's data; this category is out of scope for a backend RAG integrity layer as currently built. |
| Data retention timelines (auto-deletion after inactivity) | Gap | `src/unlearning/delete.py` supports erasure *on request*; there is no scheduled/automatic retention-expiry deletion job. |

## Summary

VeritasRAG's strongest DPDP-relevant coverage is on the **erasure and
auditability** side — Phase 6's signed deletion receipts with live-recomputed
validity and Phase 7's structured, queryable trace log are exactly the kind
of verifiable technical evidence those obligation categories call for. The
weakest coverage is on the **consent and Data-Principal-rights** side, which
is expected: this project is a RAG integrity/defense layer over a document
corpus, not an end-user-facing product with a data-collection surface, so
those categories have no natural analog to build against yet. If VeritasRAG
is ever deployed against real personal data with real data subjects, the
consent, access/correction, and breach-notification gaps above would need
to be closed before any compliance claim could be made — they are not
implicitly covered by anything documented in this file.
