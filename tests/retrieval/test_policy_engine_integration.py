import pytest
from src.embeddings.embedder import Embedder
from src.ingestion.provenance import ProvenanceSigner
from src.retrieval.config import PolicyConfig
from src.retrieval.policy_engine import FrequencyTracker, GateDecision, retrieve
from src.vectorstore.qdrant_client import ensure_collection, get_client, upsert_manifest
from src.vectorstore.schema import TrustTier

TEST_COLLECTION = "veritasrag_test_scratch_retrieval"

UNVERIFIED_MANIFEST = [
    {
        "chunk_id": "retr-clean-0",
        "text": "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "nq_open::9001",
    },
    {
        "chunk_id": "retr-clean-1",
        "text": "Mount Everest is the tallest mountain above sea level on Earth.",
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "nq_open::9002",
    },
]

QUARANTINED_MANIFEST = [
    {
        "chunk_id": "retr-quarantined-0",
        "text": "This chunk has been quarantined and must never be surfaced by retrieve().",
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "nq_open::9003",
    },
]


@pytest.fixture(scope="module")
def client():
    client = get_client()
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant is not reachable at localhost:6333: {exc}")
    yield client
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def signer(tmp_path):
    return ProvenanceSigner(tmp_path)


def test_retrieve_excludes_quarantined_and_returns_valid_shape(client, embedder, signer):
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)

    upsert_manifest(
        client,
        UNVERIFIED_MANIFEST,
        embedder,
        signer,
        collection_name=TEST_COLLECTION,
        trust_tier=TrustTier.UNVERIFIED,
    )
    upsert_manifest(
        client,
        QUARANTINED_MANIFEST,
        embedder,
        signer,
        collection_name=TEST_COLLECTION,
        trust_tier=TrustTier.QUARANTINED,
    )

    config = PolicyConfig(
        retrieval_collection_name=TEST_COLLECTION,
        retrieval_top_k=5,
        retrieval_overfetch_factor=3.0,
    )

    result = retrieve(
        "What is the tallest mountain on Earth?",
        config=config,
        embedder=embedder,
        qdrant_client_instance=client,
        tracker=FrequencyTracker(),  # fresh tracker, isolated from other tests' shared state
    )

    assert result.query == "What is the tallest mountain on Earth?"
    assert result.fetched_count > 0

    returned_chunk_ids = {gated.candidate.chunk_id for gated in result.candidates}
    assert "retr-quarantined-0" not in returned_chunk_ids

    for gated in result.candidates:
        assert gated.decision != GateDecision.DROP
        assert 0.0 <= gated.chunk_score.poison_score <= 1.0
        assert gated.adjusted_score == pytest.approx(
            gated.candidate.similarity_score * gated.policy_weight
        )


def test_retrieve_on_empty_collection_returns_empty_result(client, embedder):
    empty_collection = f"{TEST_COLLECTION}_empty"
    if client.collection_exists(empty_collection):
        client.delete_collection(empty_collection)
    ensure_collection(client, empty_collection, embedder.dimension)

    config = PolicyConfig(retrieval_collection_name=empty_collection)
    result = retrieve(
        "any query",
        config=config,
        embedder=embedder,
        qdrant_client_instance=client,
        tracker=FrequencyTracker(),
    )

    assert result.candidates == []
    assert result.fetched_count == 0
    assert result.dropped_count == 0

    client.delete_collection(empty_collection)
