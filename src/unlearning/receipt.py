"""Signed JSON deletion receipts, using the same Ed25519 setup as Phase 2's
provenance signer (`src/ingestion/provenance.py::ProvenanceSigner`) -- no new
keypair, no new signing scheme. Never hand-roll crypto (see `CLAUDE.md`).

Field set is a documented superset of plan_2.md's literal spec (`chunk_id`,
`deleted_at`, `requester`, `signature`): it also carries `point_id`,
`collection_name`, `content_hash`, `found`, and the two ghost-vector-defense
check outcomes, since an audit-grade deletion receipt needs to say not just
*that* something was deleted but *what was verified about the deletion*.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.ingestion.provenance import ProvenanceSigner
from src.unlearning.config import DEFAULT_CONFIG, UnlearningConfig

RECEIPT_FIELDS = (
    "receipt_id",
    "chunk_id",
    "point_id",
    "collection_name",
    "content_hash",
    "found",
    "deleted_at",
    "requester",
    "membership_check_passed",
    "reconstruction_check_passed",
)


def canonical_receipt_json(receipt_body: dict) -> str:
    """Deterministic JSON serialization of a receipt body.

    `sort_keys=True` and fixed separators matter: without them,
    `verify_receipt` could fail nondeterministically depending on the dict's
    insertion order or a JSON round-trip's whitespace, even though nothing
    about the receipt's content actually changed.
    """
    return json.dumps(receipt_body, sort_keys=True, separators=(",", ":"))


def build_receipt(
    *,
    chunk_id: str,
    point_id: str,
    collection_name: str,
    content_hash: str | None,
    found: bool,
    requester: str,
    deleted_at: str | None = None,
    membership_check_passed: bool | None = None,
    reconstruction_check_passed: bool | None = None,
    receipt_id: str | None = None,
) -> dict:
    """Assemble the unsigned receipt body (no `receipt_hash`/`signature` key yet)."""
    return {
        "receipt_id": receipt_id or str(uuid.uuid4()),
        "chunk_id": chunk_id,
        "point_id": point_id,
        "collection_name": collection_name,
        "content_hash": content_hash,
        "found": found,
        "deleted_at": deleted_at or datetime.now(UTC).isoformat(),
        "requester": requester,
        "membership_check_passed": membership_check_passed,
        "reconstruction_check_passed": reconstruction_check_passed,
    }


def sign_receipt(receipt_body: dict, signer: ProvenanceSigner | None = None) -> dict:
    """Hash the canonical receipt body and sign that hash with `signer`
    (defaults to `ProvenanceSigner()`, i.e. Phase 2's project-wide keypair
    under `keys/`). Returns the body plus `receipt_hash` and `signature`.
    """
    signer = signer or ProvenanceSigner()
    receipt_hash = signer.hash_content(canonical_receipt_json(receipt_body))
    signature = signer.sign(receipt_hash)
    return {**receipt_body, "receipt_hash": receipt_hash, "signature": signature}


def verify_receipt(receipt: dict, signer: ProvenanceSigner | None = None) -> bool:
    """Recompute the canonical hash over every field except `receipt_hash`/
    `signature`, confirm it matches the stored `receipt_hash`, then verify
    the Ed25519 signature over that hash. Never raises -- returns False on
    any mismatch, missing field, or invalid/malformed signature, matching
    `ProvenanceSigner.verify`'s contract.
    """
    signer = signer or ProvenanceSigner()
    try:
        body = {k: receipt[k] for k in RECEIPT_FIELDS}
        recomputed_hash = signer.hash_content(canonical_receipt_json(body))
        if recomputed_hash != receipt.get("receipt_hash"):
            return False
        return signer.verify(recomputed_hash, receipt["signature"])
    except (KeyError, TypeError):
        return False


def save_receipt(
    receipt: dict,
    path: Path | None = None,
    config: UnlearningConfig = DEFAULT_CONFIG,
) -> Path:
    """Persist a signed receipt as indented JSON. Default path:
    `<config.receipt_dir>/<chunk_id>_<receipt_id>.json`.
    """
    if path is None:
        receipt_dir = Path(config.receipt_dir)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        path = receipt_dir / f"{receipt['chunk_id']}_{receipt['receipt_id']}.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


def load_receipt(path: Path) -> dict:
    """Load a receipt JSON file back into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cmd_verify(args) -> None:  # noqa: ANN001
    receipt = load_receipt(args.path)
    valid = verify_receipt(receipt)
    status = "VALID" if valid else "INVALID"
    print(f"[{status}] receipt {receipt.get('receipt_id')} for chunk {receipt.get('chunk_id')}")
    if not valid:
        raise SystemExit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="Independently verify a receipt file")
    verify_parser.add_argument("--path", type=Path, required=True)
    verify_parser.set_defaults(func=_cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
