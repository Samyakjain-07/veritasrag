"""Qdrant collection lifecycle, batch-upsert of a Phase 1 manifest, and
signature-sample verification.

Ground-truth note: the Phase 1 manifest's `is_poisoned` label is eval-only
scaffolding used later to score the Phase 3 detector -- it is deliberately
*not* written into the vector store's payload (as `trust_tier` or otherwise).
Doing so would let the "detector" trivially read the answer it's supposed to
predict. All chunks are upserted as `TrustTier.UNVERIFIED`; downstream phases
transition trust_tier based on detection output, not manifest ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.embeddings.embedder import Embedder
from src.ingestion.provenance import ProvenanceSigner
from src.vectorstore.schema import COLLECTION_NAME, TrustTier, build_payload

load_dotenv()

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_BATCH_SIZE = 64
DEFAULT_MANIFEST_PATH = Path("eval/benchmarks/poison_manifest.json")
DEFAULT_VERIFY_SAMPLE_SIZE = 5
DEFAULT_VERIFY_SEED = 42


def get_client(url: str = DEFAULT_QDRANT_URL) -> QdrantClient:
    """Return a Qdrant client connected to `url`."""
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create `collection_name` with the given vector size if it doesn't already exist."""
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def point_id_for_chunk(chunk_id: str) -> str:
    """Deterministic UUID5 point ID so re-running the upsert is idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def source_uri_for(source_doc: str) -> str:
    """Best-effort URI describing a chunk's origin, derived from its `source_doc` tag."""
    if source_doc.startswith("nq_open::"):
        return f"hf://sentence-transformers/natural-questions#{source_doc.split('::', 1)[1]}"
    if source_doc.startswith("poisonedrag_synthetic::"):
        return f"synthetic://poisonedrag_generator#{source_doc.split('::', 1)[1]}"
    return f"unknown://{source_doc}"


def upsert_manifest(
    client: QdrantClient,
    manifest: list[dict],
    embedder: Embedder,
    signer: ProvenanceSigner,
    collection_name: str = COLLECTION_NAME,
    trust_tier: TrustTier = TrustTier.UNVERIFIED,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Embed, sign, and batch-upsert every chunk in a Phase 1 manifest.

    Returns the number of points upserted.
    """
    ensure_collection(client, collection_name, embedder.dimension)

    total = 0
    for start in range(0, len(manifest), batch_size):
        batch = manifest[start : start + batch_size]
        vectors = embedder.embed([entry["text"] for entry in batch])

        points = []
        for entry, vector in zip(batch, vectors, strict=True):
            content_hash = signer.hash_content(entry["text"])
            signature = signer.sign(content_hash)
            payload = build_payload(
                doc_id=entry["source_doc"],
                chunk_id=entry["chunk_id"],
                source_uri=source_uri_for(entry["source_doc"]),
                trust_tier=trust_tier,
                provenance_signature=signature,
                ingested_at=datetime.now(UTC).isoformat(),
                content_hash=content_hash,
                text=entry["text"],
            )
            points.append(
                models.PointStruct(
                    id=point_id_for_chunk(entry["chunk_id"]),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        total += len(points)

    return total


def _all_point_ids(client: QdrantClient, collection_name: str) -> list[str | int]:
    ids: list[str | int] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.extend(point.id for point in points)
        if next_offset is None:
            break
    return ids


def verify_sample(
    client: QdrantClient,
    signer: ProvenanceSigner,
    collection_name: str = COLLECTION_NAME,
    sample_size: int = DEFAULT_VERIFY_SAMPLE_SIZE,
    seed: int = DEFAULT_VERIFY_SEED,
) -> list[dict]:
    """Sample points from the collection and verify their stored provenance signature.

    Checks two things per sampled point: (1) the stored `provenance_signature`
    is a valid Ed25519 signature over the stored `content_hash`, and (2)
    re-hashing the stored `text` reproduces that same `content_hash` --
    catching payload tampering that a signature check alone would miss if the
    hash and text were swapped together.
    """
    all_ids = _all_point_ids(client, collection_name)
    if not all_ids:
        raise RuntimeError(f"collection '{collection_name}' is empty; nothing to verify")

    rng = random.Random(seed)
    sample_ids = rng.sample(all_ids, min(sample_size, len(all_ids)))
    records = client.retrieve(collection_name=collection_name, ids=sample_ids, with_payload=True)

    results = []
    for record in records:
        payload = record.payload or {}
        signature_valid = signer.verify(payload["content_hash"], payload["provenance_signature"])
        recomputed_hash = signer.hash_content(payload["text"])
        hash_matches = recomputed_hash == payload["content_hash"]
        results.append(
            {
                "point_id": str(record.id),
                "chunk_id": payload.get("chunk_id"),
                "signature_valid": signature_valid,
                "content_hash_matches_text": hash_matches,
            }
        )
    return results


def search(
    client: QdrantClient,
    query_vector: list[float],
    collection_name: str = COLLECTION_NAME,
    top_k: int = 10,
    trust_tiers: list[TrustTier] | None = None,
) -> list[models.ScoredPoint]:
    """Cosine-similarity search, optionally restricted to a set of trust tiers.

    `trust_tiers=None` searches every tier; passing a list applies a
    server-side `trust_tier` filter so excluded tiers never leave Qdrant.
    """
    query_filter = None
    if trust_tiers is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="trust_tier",
                    match=models.MatchAny(any=[tier.value for tier in trust_tiers]),
                )
            ]
        )
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return response.points


def sample_random(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    sample_size: int = 128,
) -> list[models.ScoredPoint]:
    """Random sample of points (with payload), for use as a stable reference set.

    Used by the Phase 4 policy engine to stabilize the cluster-anomaly
    detector's k-NN sub-signal when scoring small per-query candidate batches
    (see `src/retrieval/policy_engine.py`'s module docstring).
    """
    response = client.query_points(
        collection_name=collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        limit=sample_size,
        with_payload=True,
    )
    return response.points


def _cmd_upsert(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    client = get_client(args.qdrant_url)
    embedder = Embedder()
    signer = ProvenanceSigner()

    total = upsert_manifest(
        client,
        manifest,
        embedder,
        signer,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    print(f"Upserted {total} chunks into '{args.collection}' from {args.manifest}")


def _cmd_verify(args: argparse.Namespace) -> None:
    client = get_client(args.qdrant_url)
    signer = ProvenanceSigner()

    results = verify_sample(
        client,
        signer,
        collection_name=args.collection,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    all_ok = True
    for result in results:
        ok = result["signature_valid"] and result["content_hash_matches_text"]
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        print(
            f"[{status}] {result['chunk_id']} (point={result['point_id']}) "
            f"signature_valid={result['signature_valid']} "
            f"content_hash_matches_text={result['content_hash_matches_text']}"
        )

    if not all_ok:
        raise SystemExit(1)
    print(f"All {len(results)} sampled signatures verified OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upsert_parser = subparsers.add_parser("upsert", help="Batch-upsert a Phase 1 manifest")
    upsert_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    upsert_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    upsert_parser.set_defaults(func=_cmd_upsert)

    verify_parser = subparsers.add_parser("verify", help="Verify a sample of stored signatures")
    verify_parser.add_argument("--sample-size", type=int, default=DEFAULT_VERIFY_SAMPLE_SIZE)
    verify_parser.add_argument("--seed", type=int, default=DEFAULT_VERIFY_SEED)
    verify_parser.set_defaults(func=_cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
