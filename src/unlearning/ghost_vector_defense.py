"""Defense against post-delete embedding reconstruction ("ghost vector"
attacks): can a deleted chunk still be summoned back from, or approximated
by, what remains in the vector store after deletion?

Two complementary, non-overlapping checks live here, matching PLAN.md's split
of "ghost_vector_defense.py defends against reconstruction" from "write an
eval script that attempts reconstruction":

1. Membership-inference self-check (`verify_unrecoverable`): after a real
   Qdrant delete + index compaction, does searching with the point's own true
   pre-delete vector still surface it? This proves deletion actually took
   effect at the index level, not just a soft tombstone. It only needs the
   Qdrant client and the captured vector -- src/unlearning/delete.py must
   capture that vector *before* calling delete, since there is no way to
   recover it afterward.

2. Corpus-level reconstruction attack (`attempt_reconstruction`): even
   without the exact deleted vector, does some *surviving* chunk still leak
   equivalent content (e.g. a near-duplicate)? This is a pure function over
   plain numpy arrays -- no Qdrant client needed -- so it's identically
   reusable by both delete.py's optional post-delete self-check and
   eval/run_eval.py's aggregate `--stage unlearning` benchmark.

Chance-baseline design note (read before changing the comparison logic):
the attack statistic is a *maximum* -- the deleted vector's similarity to its
single best-matching survivor. Comparing that maximum against the *mean* of
random-pair similarities is not a fair "chance" baseline: the max of a set
reliably exceeds the mean of a random subsample of that same set even when no
real near-duplicate exists, which would make "reconstruction succeeds" read
as true almost always, regardless of ground truth. The fix used here: draw K
decoy vectors from chunks that do not share the deleted chunk's `doc_id` (so
no legitimate near-duplicate relationship -- overlapping chunk-window
siblings, or PoisonedRAG's same-target-question variants -- contaminates the
baseline), compute each decoy's *own* best-neighbor similarity against the
same surviving corpus, and compare the real attack statistic against a high
percentile (not the mean) of that decoy distribution. This is max vs.
distribution-of-maxes, the standard shadow/non-member-query construction from
the membership-inference literature -- not max vs. mean.

Scope note: both checks operate strictly at the Qdrant-index level. Neither
says anything about the embedding model having memorized the deleted text,
LLM response caches, or generation logs retaining it -- a different, harder
threat model, out of scope per PLAN.md's literal "post-delete embedding
reconstruction via nearest-neighbor search" wording.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import faiss
import numpy as np
from qdrant_client import QdrantClient, models

from src.unlearning.config import DEFAULT_CONFIG
from src.vectorstore.qdrant_client import search

DEFAULT_SEED = 42


# --- Check 1: membership-inference self-check (Qdrant-aware) ---------------


def verify_unrecoverable(
    client: QdrantClient,
    true_vector: list[float],
    deleted_chunk_id: str,
    collection_name: str,
    search_k: int = DEFAULT_CONFIG.ghost_defense_search_k,
) -> bool:
    """True iff searching `collection_name` with the deleted point's own
    pre-delete vector no longer surfaces `deleted_chunk_id` among the top
    `search_k` results.
    """
    hits = search(client, true_vector, collection_name=collection_name, top_k=search_k)
    return all(hit.payload.get("chunk_id") != deleted_chunk_id for hit in hits)


def _scroll_all_records(client: QdrantClient, collection_name: str) -> list[models.Record]:
    """Paginated with_vectors=True scroll of every point in `collection_name`.

    Mirrors src/vectorstore/qdrant_client.py::_all_point_ids's pagination
    loop, but also fetches payload+vector since the reconstruction attack
    needs both.
    """
    records: list[models.Record] = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        records.extend(batch)
        if next_offset is None:
            break
    return records


def fetch_all_vectors(
    client: QdrantClient,
    collection_name: str,
    exclude_point_id: str | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """All surviving (point_id, doc_id, vector) triples in a collection.

    Returns parallel `(point_ids, doc_ids, vectors)`. `exclude_point_id` lets
    a caller drop a point that is about to be (or was just) deleted, e.g. to
    avoid it appearing as its own "surviving neighbor" during a pre-delete
    dry run.
    """
    records = _scroll_all_records(client, collection_name)
    point_ids: list[str] = []
    doc_ids: list[str] = []
    vectors: list[list[float]] = []
    for record in records:
        if exclude_point_id is not None and str(record.id) == exclude_point_id:
            continue
        payload = record.payload or {}
        point_ids.append(str(record.id))
        doc_ids.append(payload.get("doc_id", ""))
        vectors.append(record.vector)  # type: ignore[arg-type]

    dim = len(vectors[0]) if vectors else 0
    array = np.array(vectors, dtype=np.float32).reshape(len(vectors), dim)
    return point_ids, doc_ids, array


def sample_decoy_vectors(
    client: QdrantClient,
    collection_name: str,
    exclude_doc_id: str,
    exclude_point_id: str,
    sample_size: int,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], np.ndarray]:
    """Seeded random sample of surviving vectors whose payload `doc_id`
    differs from `exclude_doc_id`. Returns `(decoy_point_ids, decoy_vectors)`.

    Same-`doc_id` chunks are legitimate potential near-duplicates
    (overlapping RecursiveCharacterTextSplitter windows, or PoisonedRAG
    variants of the same target question) and must not be used as an
    "unrelated" null decoy -- doing so would bias the chance baseline upward
    and make the defense look weaker than it is.

    Returning point IDs (not just vectors) matters: a decoy is sampled *from*
    the surviving corpus, so the caller must exclude these same points from
    whatever index it searches the decoys against -- otherwise a decoy finds
    itself as a similarity-1.0 "nearest neighbor" and the chance baseline
    saturates at 1.0 for every deletion, an artifact that makes the
    reconstruction-attack check vacuously always pass.
    """
    point_ids, doc_ids, vectors = fetch_all_vectors(
        client, collection_name, exclude_point_id=exclude_point_id
    )
    eligible = [i for i, doc_id in enumerate(doc_ids) if doc_id != exclude_doc_id]
    if not eligible:
        return [], np.empty((0, vectors.shape[1] if vectors.size else 0), dtype=np.float32)

    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(sample_size, len(eligible)))
    return [point_ids[i] for i in chosen], vectors[chosen]


# --- Check 2: corpus-level reconstruction attack (pure numpy, no Qdrant) ---


@dataclass(frozen=True)
class ReconstructionResult:
    """Outcome of one reconstruction attempt against a deleted chunk's true
    vector, plus the decoy-derived chance baseline it was judged against.
    """

    attack_similarity: float
    top_k_neighbor_indices: list[int]
    top_k_neighbor_similarities: list[float]
    chance_baseline_similarities: list[float]
    chance_baseline_mean: float
    chance_baseline_percentile_threshold: float
    exceeds_chance: bool


def _best_neighbor_similarities(index: faiss.IndexFlatIP, queries: np.ndarray) -> np.ndarray:
    """Each query's similarity to its single nearest vector in `index`."""
    similarities, _ = index.search(queries.astype(np.float32), 1)
    return similarities[:, 0]


def attempt_reconstruction(
    true_vector: np.ndarray,
    surviving_vectors: np.ndarray,
    decoy_vectors: np.ndarray,
    top_k_diagnostic: int = DEFAULT_CONFIG.reconstruction_top_k_diagnostic,
    chance_percentile: float = DEFAULT_CONFIG.reconstruction_chance_percentile,
) -> ReconstructionResult:
    """Attempt to reconstruct `true_vector` from `surviving_vectors` via
    nearest-neighbor search, and judge the attempt against a chance baseline
    built from `decoy_vectors` (see module docstring for why this must be a
    percentile of a null *distribution*, not a mean).

    Pure function -- no Qdrant client, no internal randomness (decoy
    *selection* is the caller's job, e.g. via `sample_decoy_vectors`) -- so
    it's deterministic and trivially unit-testable with fabricated arrays.
    """
    true_vector = np.asarray(true_vector, dtype=np.float32).reshape(1, -1)

    if surviving_vectors.size == 0:
        return ReconstructionResult(
            attack_similarity=0.0,
            top_k_neighbor_indices=[],
            top_k_neighbor_similarities=[],
            chance_baseline_similarities=[],
            chance_baseline_mean=0.0,
            chance_baseline_percentile_threshold=0.0,
            exceeds_chance=False,
        )

    dim = surviving_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(surviving_vectors.astype(np.float32))

    k = min(top_k_diagnostic, surviving_vectors.shape[0])
    top_similarities, top_indices = index.search(true_vector, k)
    attack_similarity = float(top_similarities[0, 0])

    if decoy_vectors.size == 0:
        chance_similarities: list[float] = []
        chance_mean = 0.0
        chance_threshold = 0.0
    else:
        chance_similarities_arr = _best_neighbor_similarities(index, decoy_vectors)
        chance_similarities = [float(v) for v in chance_similarities_arr]
        chance_mean = float(np.mean(chance_similarities_arr))
        chance_threshold = float(np.percentile(chance_similarities_arr, chance_percentile))

    return ReconstructionResult(
        attack_similarity=attack_similarity,
        top_k_neighbor_indices=[int(i) for i in top_indices[0]],
        top_k_neighbor_similarities=[float(v) for v in top_similarities[0]],
        chance_baseline_similarities=chance_similarities,
        chance_baseline_mean=chance_mean,
        chance_baseline_percentile_threshold=chance_threshold,
        exceeds_chance=attack_similarity > chance_threshold,
    )
