"""Qdrant collection schema: trust tiers and payload fields for VeritasRAG chunks."""

from __future__ import annotations

from enum import StrEnum

COLLECTION_NAME = "veritasrag_chunks"

# Phase 8: default tenant for every chunk upserted without an explicit
# tenant_id -- keeps every pre-Phase-8 call site (and the live 280-chunk
# corpus, until re-upserted) working unmodified under a single implicit
# tenant, rather than requiring tenant_id everywhere immediately.
DEFAULT_TENANT_ID = "default"

# Phase 2 schema fields (provenance/trust metadata), per PLAN.md Phase 2.
# tenant_id added in Phase 8 for multi-tenancy isolation.
PAYLOAD_FIELDS = (
    "doc_id",
    "chunk_id",
    "source_uri",
    "trust_tier",
    "provenance_signature",
    "ingested_at",
    "content_hash",
    "tenant_id",
)


class TrustTier(StrEnum):
    """Provenance trust classification for an ingested chunk (Part 2 architecture).

    This is independent of any poison-detection score: a chunk's trust_tier
    reflects how much its *source* is trusted, not whether Phase 3/4 flagged
    its *content*. Everything lands here as UNVERIFIED at ingest time -- no
    detector exists yet at Phase 2 -- and Phase 4's policy engine is what
    later reads poison scores to quarantine or down-weight chunks.
    """

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    QUARANTINED = "quarantined"


def build_payload(
    *,
    doc_id: str,
    chunk_id: str,
    source_uri: str,
    trust_tier: TrustTier | str,
    provenance_signature: str,
    ingested_at: str,
    content_hash: str,
    text: str,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> dict:
    """Assemble a Qdrant point payload for a chunk.

    Includes `text` alongside the Phase 2 provenance/trust schema fields
    (`PAYLOAD_FIELDS`) -- retrieval and generation in later phases need the
    actual chunk content, not just its provenance metadata. `tenant_id`
    (Phase 8) defaults to `DEFAULT_TENANT_ID` so every pre-Phase-8 call site
    keeps working unmodified.
    """
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "source_uri": source_uri,
        "trust_tier": trust_tier.value if isinstance(trust_tier, TrustTier) else trust_tier,
        "provenance_signature": provenance_signature,
        "ingested_at": ingested_at,
        "content_hash": content_hash,
        "text": text,
        "tenant_id": tenant_id,
    }
