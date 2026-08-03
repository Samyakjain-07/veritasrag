import pytest
from src.embeddings.embedder import Embedder
from src.ingestion.provenance import ProvenanceSigner
from src.unlearning.config import UnlearningConfig
from src.unlearning.delete import capture_point, delete_chunk, delete_with_receipt, force_compaction
from src.unlearning.ghost_vector_defense import sample_decoy_vectors, verify_unrecoverable
from src.vectorstore.qdrant_client import get_client, point_id_for_chunk, upsert_manifest

TEST_COLLECTION = "veritasrag_unlearning_test_scratch"

SAMPLE_MANIFEST = [
    {
        "chunk_id": "chunk-a1",
        "text": "alpha document about cats and their sleeping habits",
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "docA",
    },
    {
        "chunk_id": "chunk-a2",
        "text": "alpha document part two, still about cats",
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "docA",
    },
    {
        "chunk_id": "chunk-b1",
        "text": "beta document about dogs and their loyalty",
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "docB",
    },
    {
        "chunk_id": "chunk-c1",
        "text": "gamma document about birds migrating south",
        "is_poisoned": False,
        "target_question": None,
        "source_doc": "docC",
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


def _seed_collection(client, embedder, signer) -> None:
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)
    upsert_manifest(client, SAMPLE_MANIFEST, embedder, signer, collection_name=TEST_COLLECTION)


def test_capture_point_returns_none_for_absent_chunk(client, embedder, signer):
    _seed_collection(client, embedder, signer)

    assert capture_point(client, "never-existed", TEST_COLLECTION) is None


def test_capture_point_returns_payload_and_vector(client, embedder, signer):
    _seed_collection(client, embedder, signer)

    record = capture_point(client, "chunk-a1", TEST_COLLECTION)

    assert record is not None
    assert record.payload["chunk_id"] == "chunk-a1"
    assert record.vector is not None


def test_delete_chunk_removes_only_target(client, embedder, signer):
    _seed_collection(client, embedder, signer)

    result = delete_chunk(client, "chunk-a1", TEST_COLLECTION)

    assert result.found is True
    assert capture_point(client, "chunk-a1", TEST_COLLECTION) is None
    assert capture_point(client, "chunk-a2", TEST_COLLECTION) is not None
    assert capture_point(client, "chunk-b1", TEST_COLLECTION) is not None


def test_delete_chunk_is_noop_on_absent_chunk(client, embedder, signer):
    _seed_collection(client, embedder, signer)

    result = delete_chunk(client, "does-not-exist", TEST_COLLECTION)

    assert result.found is False
    assert result.delete_latency_ms == 0.0
    assert result.compaction_triggered is False


def test_force_compaction_restores_original_thresholds(client, embedder, signer):
    _seed_collection(client, embedder, signer)
    original = client.get_collection(TEST_COLLECTION).config.optimizer_config

    triggered, latency_ms, status = force_compaction(client, TEST_COLLECTION)

    assert triggered is True
    assert latency_ms >= 0.0
    restored = client.get_collection(TEST_COLLECTION).config.optimizer_config
    assert restored.deleted_threshold == original.deleted_threshold
    assert restored.vacuum_min_vector_number == original.vacuum_min_vector_number


def test_verify_unrecoverable_before_and_after_delete(client, embedder, signer):
    _seed_collection(client, embedder, signer)
    record = capture_point(client, "chunk-a1", TEST_COLLECTION)

    # Negative control: still present, so the true vector must still surface it.
    assert verify_unrecoverable(client, record.vector, "chunk-a1", TEST_COLLECTION) is False

    delete_chunk(client, "chunk-a1", TEST_COLLECTION)

    assert verify_unrecoverable(client, record.vector, "chunk-a1", TEST_COLLECTION) is True


def test_sample_decoy_vectors_excludes_same_doc_id(client, embedder, signer):
    _seed_collection(client, embedder, signer)

    decoy_point_ids, decoy_vectors = sample_decoy_vectors(
        client,
        TEST_COLLECTION,
        exclude_doc_id="docA",
        exclude_point_id=point_id_for_chunk("chunk-a1"),
        sample_size=10,
        seed=1,
    )

    # 4 chunks total; excluding chunk-a1 by point_id and chunk-a2 by doc_id
    # leaves only chunk-b1/chunk-c1 eligible.
    assert len(decoy_point_ids) == 2
    assert decoy_vectors.shape[0] == 2
    assert point_id_for_chunk("chunk-a1") not in decoy_point_ids
    assert point_id_for_chunk("chunk-a2") not in decoy_point_ids


def test_delete_with_receipt_end_to_end_found(client, embedder, signer, tmp_path):
    _seed_collection(client, embedder, signer)
    config = UnlearningConfig(receipt_dir=str(tmp_path / "receipts"))

    bundle = delete_with_receipt(
        client,
        "chunk-a1",
        requester="tester",
        collection_name=TEST_COLLECTION,
        config=config,
        signer=signer,
    )

    assert bundle.deletion.found is True
    assert bundle.membership_check_passed is True
    assert bundle.reconstruction_check is not None
    # Regression guard: decoys must never be able to find themselves in the
    # index they're searched against (a self-match artifact that would
    # saturate the chance baseline at a trivial 1.0 for every deletion).
    assert bundle.reconstruction_check.chance_baseline_mean < 0.999
    assert bundle.receipt_valid is True
    assert bundle.receipt_path.exists()


def test_delete_with_receipt_end_to_end_not_found(client, embedder, signer, tmp_path):
    _seed_collection(client, embedder, signer)
    config = UnlearningConfig(receipt_dir=str(tmp_path / "receipts_not_found"))

    bundle = delete_with_receipt(
        client,
        "never-existed",
        requester="tester",
        collection_name=TEST_COLLECTION,
        config=config,
        signer=signer,
    )

    assert bundle.deletion.found is False
    assert bundle.membership_check_passed is None
    assert bundle.reconstruction_check is None
    assert bundle.receipt["found"] is False
    assert bundle.receipt_valid is True
