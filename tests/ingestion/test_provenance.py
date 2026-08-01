from src.ingestion.provenance import (
    ProvenanceSigner,
    compute_content_hash,
    load_or_create_keypair,
    sign_content_hash,
    verify_signature,
)


def test_compute_content_hash_is_deterministic():
    assert compute_content_hash("hello world") == compute_content_hash("hello world")
    assert compute_content_hash("hello world") != compute_content_hash("hello world!")


def test_sign_and_verify_roundtrip(tmp_path):
    private_key, public_key = load_or_create_keypair(tmp_path)
    content_hash = compute_content_hash("some chunk text")

    signature = sign_content_hash(private_key, content_hash)

    assert verify_signature(public_key, content_hash, signature) is True


def test_verify_rejects_tampered_hash(tmp_path):
    private_key, public_key = load_or_create_keypair(tmp_path)
    signature = sign_content_hash(private_key, compute_content_hash("original text"))

    assert verify_signature(public_key, compute_content_hash("different text"), signature) is False


def test_verify_rejects_malformed_signature_without_raising(tmp_path):
    _, public_key = load_or_create_keypair(tmp_path)

    assert verify_signature(public_key, "deadbeef", "not-valid-base64!!") is False


def test_load_or_create_keypair_is_persisted(tmp_path):
    private_key_1, public_key_1 = load_or_create_keypair(tmp_path)
    private_key_2, public_key_2 = load_or_create_keypair(tmp_path)

    assert public_key_1.public_bytes_raw() == public_key_2.public_bytes_raw()
    assert private_key_1.private_bytes_raw() == private_key_2.private_bytes_raw()


def test_provenance_signer_hash_sign_verify(tmp_path):
    signer = ProvenanceSigner(tmp_path)
    content_hash = signer.hash_content("chunk text")
    signature = signer.sign(content_hash)

    assert signer.verify(content_hash, signature) is True
    assert signer.verify(signer.hash_content("other text"), signature) is False
