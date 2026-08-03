"""Delete-by-ID against Qdrant with real index compaction, orchestrating the
Phase 6 unlearning flow end to end: capture -> delete -> compact -> ghost-
vector defense checks -> signed receipt.

The CLI's default target is the *live* collection (matching Phase 2's
`src/vectorstore/qdrant_client.py` upsert/verify CLI's own default-to-live-
collection convention). Running `python -m src.unlearning.delete --chunk-id
<id>` for real removes that chunk from the live demo corpus permanently --
intentional per PLAN.md's literal verify command, but worth being deliberate
about which chunk_id you pass. `eval/run_eval.py --stage unlearning` never
touches the live collection; it always targets its own disposable scratch
collection.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.ingestion.provenance import ProvenanceSigner
from src.unlearning.config import DEFAULT_CONFIG, UnlearningConfig
from src.unlearning.ghost_vector_defense import (
    ReconstructionResult,
    attempt_reconstruction,
    fetch_all_vectors,
    sample_decoy_vectors,
    verify_unrecoverable,
)
from src.unlearning.receipt import build_receipt, save_receipt, sign_receipt, verify_receipt
from src.vectorstore.qdrant_client import DEFAULT_QDRANT_URL, get_client, point_id_for_chunk

load_dotenv()


@dataclass(frozen=True)
class DeletionResult:
    """Outcome of a single delete-by-ID call against Qdrant."""

    chunk_id: str
    point_id: str
    collection_name: str
    found: bool
    content_hash: str | None
    pre_delete_payload: dict | None
    pre_delete_vector: list[float] | None
    deleted_at: str
    delete_latency_ms: float
    compaction_triggered: bool
    compaction_latency_ms: float | None
    compaction_status: str | None


@dataclass(frozen=True)
class DeletionReceiptBundle:
    """Full Phase 6 outcome: the deletion itself, both ghost-vector-defense
    checks, and the signed receipt attesting to all of it.
    """

    deletion: DeletionResult
    membership_check_passed: bool | None
    reconstruction_check: ReconstructionResult | None
    receipt: dict
    receipt_path: Path
    receipt_gen_latency_ms: float
    receipt_verify_latency_ms: float
    receipt_valid: bool


def capture_point(
    client: QdrantClient, chunk_id: str, collection_name: str
) -> models.Record | None:
    """Fetch a point's payload + vector before deleting it.

    Must run strictly *before* `client.delete` -- once a point is deleted
    there is no way to recover its vector, which both the receipt's
    `content_hash` and the ghost-vector-defense checks depend on.
    """
    point_id = point_id_for_chunk(chunk_id)
    records = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )
    return records[0] if records else None


def force_compaction(
    client: QdrantClient,
    collection_name: str,
    config: UnlearningConfig = DEFAULT_CONFIG,
) -> tuple[bool, float, str | None]:
    """Force Qdrant to immediately vacuum any segment holding a just-deleted
    point, rather than waiting for the server's default
    deleted_threshold=0.2 / vacuum_min_vector_number=1000 to be crossed
    naturally.

    Snapshots the collection's current optimizer thresholds first and
    restores them afterward (if `config.compaction_restore_original_config`)
    -- `update_collection` is a standing config change, not a one-shot
    action, so without the restore step a single delete call would
    permanently change the collection's optimizer behavior for every future
    write.

    Returns `(triggered, latency_ms, final_status_or_None)`.
    """
    start = time.perf_counter()

    original = client.get_collection(collection_name).config.optimizer_config

    client.update_collection(
        collection_name=collection_name,
        optimizers_config=models.OptimizersConfigDiff(
            deleted_threshold=config.compaction_deleted_threshold,
            vacuum_min_vector_number=config.compaction_vacuum_min_vector_number,
        ),
    )

    final_status: str | None = None
    if config.compaction_poll_for_green:
        deadline = time.perf_counter() + config.compaction_poll_timeout_s
        while time.perf_counter() < deadline:
            status = client.get_collection(collection_name).status
            final_status = status.value
            if status == models.CollectionStatus.GREEN:
                break
            time.sleep(config.compaction_poll_interval_s)

    if config.compaction_restore_original_config:
        client.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                deleted_threshold=original.deleted_threshold,
                vacuum_min_vector_number=original.vacuum_min_vector_number,
            ),
        )

    latency_ms = (time.perf_counter() - start) * 1000
    return True, latency_ms, final_status


def delete_chunk(
    client: QdrantClient,
    chunk_id: str,
    collection_name: str | None = None,
    config: UnlearningConfig = DEFAULT_CONFIG,
    compact: bool = True,
) -> DeletionResult:
    """Delete a single chunk by ID, with index compaction.

    Idempotent: deleting an already-absent (or never-upserted) `chunk_id` is
    a legitimate no-op (`found=False`), not an error -- no delete/compaction
    calls are made in that case.
    """
    collection_name = collection_name or config.unlearning_collection_name
    point_id = point_id_for_chunk(chunk_id)

    record = capture_point(client, chunk_id, collection_name)
    if record is None:
        return DeletionResult(
            chunk_id=chunk_id,
            point_id=point_id,
            collection_name=collection_name,
            found=False,
            content_hash=None,
            pre_delete_payload=None,
            pre_delete_vector=None,
            deleted_at=datetime.now(UTC).isoformat(),
            delete_latency_ms=0.0,
            compaction_triggered=False,
            compaction_latency_ms=None,
            compaction_status=None,
        )

    payload = record.payload or {}

    start = time.perf_counter()
    client.delete(
        collection_name=collection_name,
        points_selector=[point_id],
        wait=config.delete_wait,
    )
    delete_latency_ms = (time.perf_counter() - start) * 1000

    compaction_triggered = False
    compaction_latency_ms: float | None = None
    compaction_status: str | None = None
    if compact:
        compaction_triggered, compaction_latency_ms, compaction_status = force_compaction(
            client, collection_name, config
        )

    return DeletionResult(
        chunk_id=chunk_id,
        point_id=point_id,
        collection_name=collection_name,
        found=True,
        content_hash=payload.get("content_hash"),
        pre_delete_payload=payload,
        pre_delete_vector=record.vector,  # type: ignore[arg-type]
        deleted_at=datetime.now(UTC).isoformat(),
        delete_latency_ms=delete_latency_ms,
        compaction_triggered=compaction_triggered,
        compaction_latency_ms=compaction_latency_ms,
        compaction_status=compaction_status,
    )


def delete_with_receipt(
    client: QdrantClient,
    chunk_id: str,
    *,
    requester: str,
    collection_name: str | None = None,
    config: UnlearningConfig = DEFAULT_CONFIG,
    signer: ProvenanceSigner | None = None,
    compact: bool = True,
    run_reconstruction_check: bool = True,
    decoy_seed: int = 42,
) -> DeletionReceiptBundle:
    """Full Phase 6 flow: delete -> ghost-vector-defense checks -> signed
    receipt. Called by both the CLI (`main`, below) and
    `eval/run_eval.py::run_unlearning_eval` so the two can never drift out of
    sync.
    """
    collection_name = collection_name or config.unlearning_collection_name
    signer = signer or ProvenanceSigner()

    deletion = delete_chunk(client, chunk_id, collection_name, config, compact=compact)

    membership_check_passed: bool | None = None
    reconstruction_check: ReconstructionResult | None = None

    if deletion.found:
        membership_check_passed = verify_unrecoverable(
            client,
            true_vector=deletion.pre_delete_vector,
            deleted_chunk_id=chunk_id,
            collection_name=collection_name,
            search_k=config.ghost_defense_search_k,
        )

        if run_reconstruction_check:
            doc_id = (deletion.pre_delete_payload or {}).get("doc_id", "")
            decoy_point_ids, decoy_vectors = sample_decoy_vectors(
                client,
                collection_name,
                exclude_doc_id=doc_id,
                exclude_point_id=deletion.point_id,
                sample_size=config.reconstruction_chance_sample_size,
                seed=decoy_seed,
            )
            # The decoys must be excluded from the index they're searched
            # against -- otherwise a decoy trivially "finds itself" as a
            # similarity-1.0 nearest neighbor, saturating the chance
            # baseline and making the check vacuously always pass. See
            # sample_decoy_vectors's docstring.
            survivor_point_ids, _, all_vectors = fetch_all_vectors(client, collection_name)
            excluded = set(decoy_point_ids)
            index_mask = [pid not in excluded for pid in survivor_point_ids]
            surviving_vectors = all_vectors[index_mask]
            reconstruction_check = attempt_reconstruction(
                true_vector=deletion.pre_delete_vector,
                surviving_vectors=surviving_vectors,
                decoy_vectors=decoy_vectors,
                top_k_diagnostic=config.reconstruction_top_k_diagnostic,
                chance_percentile=config.reconstruction_chance_percentile,
            )

    receipt_start = time.perf_counter()
    receipt_body = build_receipt(
        chunk_id=deletion.chunk_id,
        point_id=deletion.point_id,
        collection_name=deletion.collection_name,
        content_hash=deletion.content_hash,
        found=deletion.found,
        requester=requester,
        deleted_at=deletion.deleted_at,
        membership_check_passed=membership_check_passed,
        reconstruction_check_passed=(
            not reconstruction_check.exceeds_chance if reconstruction_check else None
        ),
    )
    receipt = sign_receipt(receipt_body, signer)
    receipt_path = save_receipt(receipt, config=config)
    receipt_gen_latency_ms = (time.perf_counter() - receipt_start) * 1000

    verify_start = time.perf_counter()
    receipt_valid = verify_receipt(receipt, signer)
    receipt_verify_latency_ms = (time.perf_counter() - verify_start) * 1000

    return DeletionReceiptBundle(
        deletion=deletion,
        membership_check_passed=membership_check_passed,
        reconstruction_check=reconstruction_check,
        receipt=receipt,
        receipt_path=receipt_path,
        receipt_gen_latency_ms=receipt_gen_latency_ms,
        receipt_verify_latency_ms=receipt_verify_latency_ms,
        receipt_valid=receipt_valid,
    )


def _cmd_delete(args: argparse.Namespace) -> None:
    client = get_client(args.qdrant_url)
    config = DEFAULT_CONFIG
    collection_name = args.collection or config.unlearning_collection_name

    bundle = delete_with_receipt(
        client,
        args.chunk_id,
        requester=args.requester,
        collection_name=collection_name,
        run_reconstruction_check=not args.skip_reconstruction_check,
    )

    deletion = bundle.deletion
    if not deletion.found:
        print(f"chunk '{args.chunk_id}' not found in '{collection_name}' -- nothing to delete.")
        print(f"Receipt (found=False) written to {bundle.receipt_path}")
        return

    print(f"Deleted chunk '{deletion.chunk_id}' (point={deletion.point_id})")
    print(f"  delete latency:      {deletion.delete_latency_ms:.1f} ms")
    print(f"  compaction latency:  {deletion.compaction_latency_ms:.1f} ms")
    print(f"  compaction status:   {deletion.compaction_status}")
    print(f"  membership check:    {'PASSED' if bundle.membership_check_passed else 'FAILED'}")
    if bundle.reconstruction_check is not None:
        rc = bundle.reconstruction_check
        outcome = "no leak detected" if not rc.exceeds_chance else "possible leak detected"
        print(
            f"  reconstruction check: {outcome} "
            f"(similarity={rc.attack_similarity:.3f}, "
            f"chance_threshold={rc.chance_baseline_percentile_threshold:.3f})"
        )
    print(f"  receipt:             {bundle.receipt_path} (valid={bundle.receipt_valid})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument(
        "--collection",
        default=None,
        help="Defaults to UnlearningConfig.unlearning_collection_name (the live collection).",
    )
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument("--requester", default=os.getenv("USER", "unspecified"))
    parser.add_argument(
        "--skip-reconstruction-check",
        action="store_true",
        help="Skip the corpus-level reconstruction attack (still runs the membership check).",
    )
    parser.set_defaults(func=_cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
