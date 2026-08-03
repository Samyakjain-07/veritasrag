from src.ingestion.provenance import ProvenanceSigner
from src.unlearning.config import UnlearningConfig
from src.unlearning.receipt import (
    RECEIPT_FIELDS,
    build_receipt,
    canonical_receipt_json,
    load_receipt,
    save_receipt,
    sign_receipt,
    verify_receipt,
)


def _build(**overrides) -> dict:
    fields = {
        "chunk_id": "clean-00001-0",
        "point_id": "1234",
        "collection_name": "veritasrag_chunks",
        "content_hash": "deadbeef",
        "found": True,
        "requester": "test-user",
    }
    fields.update(overrides)
    return build_receipt(**fields)


def test_build_receipt_contains_required_fields():
    receipt = _build()
    assert set(RECEIPT_FIELDS) <= set(receipt.keys())
    assert receipt["chunk_id"] == "clean-00001-0"
    assert receipt["requester"] == "test-user"
    assert receipt["deleted_at"]


def test_sign_and_verify_roundtrip(tmp_path):
    signer = ProvenanceSigner(tmp_path)
    receipt = sign_receipt(_build(), signer)

    assert "receipt_hash" in receipt
    assert "signature" in receipt
    assert verify_receipt(receipt, signer) is True


def test_verify_rejects_tampered_field(tmp_path):
    signer = ProvenanceSigner(tmp_path)
    receipt = sign_receipt(_build(), signer)

    tampered = {**receipt, "requester": "attacker"}

    assert verify_receipt(tampered, signer) is False


def test_verify_rejects_wrong_signer(tmp_path):
    signer_a = ProvenanceSigner(tmp_path / "a")
    signer_b = ProvenanceSigner(tmp_path / "b")
    receipt = sign_receipt(_build(), signer_a)

    assert verify_receipt(receipt, signer_b) is False


def test_verify_handles_missing_fields_without_raising(tmp_path):
    signer = ProvenanceSigner(tmp_path)
    assert verify_receipt({}, signer) is False
    assert verify_receipt({"receipt_hash": "x", "signature": "y"}, signer) is False


def test_canonical_receipt_json_is_insertion_order_independent():
    body_a = {"a": 1, "b": 2}
    body_b = {"b": 2, "a": 1}

    assert canonical_receipt_json(body_a) == canonical_receipt_json(body_b)


def test_save_and_load_roundtrip(tmp_path):
    signer = ProvenanceSigner(tmp_path)
    receipt = sign_receipt(_build(), signer)
    config = UnlearningConfig(receipt_dir=str(tmp_path / "receipts"))

    path = save_receipt(receipt, config=config)
    loaded = load_receipt(path)

    assert loaded == receipt
    assert verify_receipt(loaded, signer) is True
