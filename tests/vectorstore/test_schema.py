from src.vectorstore.schema import DEFAULT_TENANT_ID, PAYLOAD_FIELDS, TrustTier, build_payload


def test_trust_tier_values():
    assert TrustTier.VERIFIED == "verified"
    assert TrustTier.UNVERIFIED == "unverified"
    assert TrustTier.QUARANTINED == "quarantined"


def test_build_payload_contains_schema_fields_and_text():
    payload = build_payload(
        doc_id="doc-1",
        chunk_id="chunk-1",
        source_uri="hf://example#1",
        trust_tier=TrustTier.UNVERIFIED,
        provenance_signature="sig",
        ingested_at="2026-01-01T00:00:00+00:00",
        content_hash="abc123",
        text="the chunk text",
    )

    assert set(PAYLOAD_FIELDS) <= payload.keys()
    assert payload["trust_tier"] == "unverified"
    assert payload["text"] == "the chunk text"
    assert payload["tenant_id"] == DEFAULT_TENANT_ID


def test_build_payload_accepts_plain_string_trust_tier():
    payload = build_payload(
        doc_id="doc-1",
        chunk_id="chunk-1",
        source_uri="hf://example#1",
        trust_tier="quarantined",
        provenance_signature="sig",
        ingested_at="2026-01-01T00:00:00+00:00",
        content_hash="abc123",
        text="text",
    )

    assert payload["trust_tier"] == "quarantined"


def test_build_payload_accepts_explicit_tenant_id():
    payload = build_payload(
        doc_id="doc-1",
        chunk_id="chunk-1",
        source_uri="hf://example#1",
        trust_tier=TrustTier.UNVERIFIED,
        provenance_signature="sig",
        ingested_at="2026-01-01T00:00:00+00:00",
        content_hash="abc123",
        text="text",
        tenant_id="tenant-a",
    )

    assert payload["tenant_id"] == "tenant-a"
