"""Ed25519 content-hash signing for chunk provenance.

All hashing and signing goes through the `cryptography` package -- never
hand-roll crypto (see `CLAUDE.md`).

Design note: this issues one project-wide signing keypair (the "ingestion
gateway" identity), not a per-chunk or per-source key -- matching the
architecture's single provenance signer. The keypair is persisted under
`keys/` (gitignored, unencrypted PKCS8 PEM -- acceptable for local dev, not
for production key custody) so verification works across process restarts.
Set `PROVENANCE_KEY_DIR` to relocate it.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from dotenv import load_dotenv

load_dotenv()

DEFAULT_KEY_DIR = Path(os.getenv("PROVENANCE_KEY_DIR", "keys"))
PRIVATE_KEY_FILENAME = "provenance_ed25519_private.pem"
PUBLIC_KEY_FILENAME = "provenance_ed25519_public.pem"


def compute_content_hash(text: str) -> str:
    """SHA-256 hash of `text`, hex-encoded, computed via `cryptography`'s hash API."""
    digest = hashes.Hash(hashes.SHA256())
    digest.update(text.encode("utf-8"))
    return digest.finalize().hex()


def load_or_create_keypair(
    key_dir: Path = DEFAULT_KEY_DIR,
) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Load the project's Ed25519 signing keypair from `key_dir`, generating one if absent."""
    key_dir.mkdir(parents=True, exist_ok=True)
    private_path = key_dir / PRIVATE_KEY_FILENAME
    public_path = key_dir / PUBLIC_KEY_FILENAME

    if private_path.exists():
        private_key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError(f"{private_path} does not contain an Ed25519 private key")
        return private_key, private_key.public_key()

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_key, public_key


def load_public_key(key_dir: Path = DEFAULT_KEY_DIR) -> Ed25519PublicKey:
    """Load just the public key, for verification-only callers."""
    public_path = key_dir / PUBLIC_KEY_FILENAME
    public_key = serialization.load_pem_public_key(public_path.read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError(f"{public_path} does not contain an Ed25519 public key")
    return public_key


def sign_content_hash(private_key: Ed25519PrivateKey, content_hash: str) -> str:
    """Sign a hex content hash, returning a base64-encoded signature."""
    signature = private_key.sign(content_hash.encode("utf-8"))
    return base64.b64encode(signature).decode("ascii")


def verify_signature(public_key: Ed25519PublicKey, content_hash: str, signature_b64: str) -> bool:
    """Verify a base64-encoded signature over a hex content hash.

    Returns False (never raises) for both an invalid signature and malformed
    signature data -- a tampered/corrupted payload should fail verification
    cleanly rather than crash the caller.
    """
    try:
        signature = base64.b64decode(signature_b64, validate=True)
        public_key.verify(signature, content_hash.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError):
        return False


class ProvenanceSigner:
    """Bundles the project's loaded Ed25519 keypair with hash/sign/verify helpers."""

    def __init__(self, key_dir: Path = DEFAULT_KEY_DIR) -> None:
        self.private_key, self.public_key = load_or_create_keypair(key_dir)

    def hash_content(self, text: str) -> str:
        """SHA-256 hex digest of `text`."""
        return compute_content_hash(text)

    def sign(self, content_hash: str) -> str:
        """Sign `content_hash` with this signer's private key."""
        return sign_content_hash(self.private_key, content_hash)

    def verify(self, content_hash: str, signature_b64: str) -> bool:
        """Verify `signature_b64` over `content_hash` against this signer's public key."""
        return verify_signature(self.public_key, content_hash, signature_b64)
