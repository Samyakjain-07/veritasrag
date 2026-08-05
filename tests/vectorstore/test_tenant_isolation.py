"""Phase 8 multi-tenancy isolation tests: confirm `tenant_id` filtering on
`search()`/`retrieve()` never leaks one tenant's chunks into another's
results, and that omitting `tenant_id` stays unfiltered (backward-compatible
with every pre-Phase-8 call site).

Mirrors `tests/retrieval/test_policy_engine_integration.py`'s exact
module-scoped-client/skip-if-Qdrant-unreachable/teardown fixture pattern.

Both tenants' manifests use the *same* fact text under distinct chunk_id
namespaces (`tenant-a-*` / `tenant-b-*`) -- the strongest possible test: with
identical text, an unfiltered search would score both tenants' chunks
equally, so any observed exclusion must come from the tenant_id filter
actually working, not incidental semantic dissimilarity. Distinct chunk_id
namespaces also sidestep a known, deliberately-unfixed gap this phase
documents: `point_id_for_chunk()` derives a point's Qdrant ID from `chunk_id`
alone, not `(tenant_id, chunk_id)`, so two tenants sharing a chunk_id would
silently overwrite each other's point.
"""

import pytest
from src.embeddings.embedder import Embedder
from src.ingestion.provenance import ProvenanceSigner
from src.retrieval.config import PolicyConfig
from src.retrieval.policy_engine import FrequencyTracker, retrieve
from src.vectorstore.qdrant_client import get_client, search, upsert_manifest

TEST_COLLECTION = "veritasrag_test_scratch_tenant_isolation"

SHARED_FACT_TEXT = "Acme Corp's Q3 revenue was $4.2 million, up from $3.1 million a year earlier."

TENANT_A_MANIFEST = [
    {
        "chunk_id": "tenant-a-chunk-0",
        "text": SHARED_FACT_TEXT,
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "nq_open::8001",
    }
]

TENANT_B_MANIFEST = [
    {
        "chunk_id": "tenant-b-chunk-0",
        "text": SHARED_FACT_TEXT,
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "nq_open::8002",
    }
]

QUERY = "What was Acme Corp's Q3 revenue?"


@pytest.fixture(scope="module")
def client():
    client = get_client()
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant is not reachable at localhost:6333: {exc}")
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)

    embedder = Embedder()
    signer = ProvenanceSigner()
    upsert_manifest(
        client,
        TENANT_A_MANIFEST,
        embedder,
        signer,
        collection_name=TEST_COLLECTION,
        tenant_id="tenant-a",
    )
    upsert_manifest(
        client,
        TENANT_B_MANIFEST,
        embedder,
        signer,
        collection_name=TEST_COLLECTION,
        tenant_id="tenant-b",
    )

    yield client
    client.delete_collection(TEST_COLLECTION)


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_tenant_isolation_search_excludes_other_tenant(client, embedder):
    query_vector = embedder.embed_one(QUERY).tolist()

    tenant_a_hits = search(
        client, query_vector, collection_name=TEST_COLLECTION, top_k=10, tenant_id="tenant-a"
    )
    tenant_a_chunk_ids = {hit.payload["chunk_id"] for hit in tenant_a_hits}
    assert "tenant-a-chunk-0" in tenant_a_chunk_ids
    assert "tenant-b-chunk-0" not in tenant_a_chunk_ids

    tenant_b_hits = search(
        client, query_vector, collection_name=TEST_COLLECTION, top_k=10, tenant_id="tenant-b"
    )
    tenant_b_chunk_ids = {hit.payload["chunk_id"] for hit in tenant_b_hits}
    assert "tenant-b-chunk-0" in tenant_b_chunk_ids
    assert "tenant-a-chunk-0" not in tenant_b_chunk_ids


def test_tenant_isolation_retrieve_excludes_other_tenant(client, embedder):
    config = PolicyConfig(retrieval_collection_name=TEST_COLLECTION, retrieval_top_k=10)

    result_a = retrieve(
        QUERY,
        config=config,
        embedder=embedder,
        qdrant_client_instance=client,
        tracker=FrequencyTracker(),
        tenant_id="tenant-a",
    )
    chunk_ids_a = {gated.candidate.chunk_id for gated in result_a.candidates}
    assert "tenant-a-chunk-0" in chunk_ids_a
    assert "tenant-b-chunk-0" not in chunk_ids_a

    result_b = retrieve(
        QUERY,
        config=config,
        embedder=embedder,
        qdrant_client_instance=client,
        tracker=FrequencyTracker(),
        tenant_id="tenant-b",
    )
    chunk_ids_b = {gated.candidate.chunk_id for gated in result_b.candidates}
    assert "tenant-b-chunk-0" in chunk_ids_b
    assert "tenant-a-chunk-0" not in chunk_ids_b


def test_tenant_isolation_default_is_unfiltered_across_tenants(client, embedder):
    query_vector = embedder.embed_one(QUERY).tolist()

    hits = search(client, query_vector, collection_name=TEST_COLLECTION, top_k=10)
    chunk_ids = {hit.payload["chunk_id"] for hit in hits}
    assert "tenant-a-chunk-0" in chunk_ids
    assert "tenant-b-chunk-0" in chunk_ids

    config = PolicyConfig(retrieval_collection_name=TEST_COLLECTION, retrieval_top_k=10)
    result = retrieve(
        QUERY,
        config=config,
        embedder=embedder,
        qdrant_client_instance=client,
        tracker=FrequencyTracker(),
    )
    retrieved_chunk_ids = {gated.candidate.chunk_id for gated in result.candidates}
    assert "tenant-a-chunk-0" in retrieved_chunk_ids
    assert "tenant-b-chunk-0" in retrieved_chunk_ids
