"""Phase 4 retrieval policy engine: gates Qdrant search results on trust_tier
and a live-computed poison_score, flags retrieval-frequency anomalies, and
exposes `retrieve(query)` core logic plus a thin LangGraph node adapter.

`poison_score` is never stored in Qdrant (Phase 2/3 deliberately keep it out
of the payload to avoid label leakage), so it's computed fresh per call via
`DetectionPipeline.score_chunks()` over the retrieved candidate batch.

Known limitation, preserved here rather than silently worked around:
`score_cluster_anomaly` (src/detection/cluster_anomaly.py) min-max normalizes
its `cluster_anomaly_norm` sub-score over whatever batch it's given. Passing
`config.cluster_reference_sample_size` chunks as `DetectionPipeline.
score_chunks`'s `reference_texts` improves the k-NN density sub-signal's
comparability across calls, but the DBSCAN sub-signal stays self-referential
by construction and the *final* normalization is always computed over the
small per-query candidate batch regardless of any reference set -- so the
classifier feature trained on a full-280-chunk-batch-normalized score is
still evaluated here at a different scale than it was trained on. This is a
partial mitigation, not a fix; a full fix needs `cluster_anomaly.py` to
support a fixed/anchored normalization range (Phase 9 territory).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from math import ceil
from statistics import fmean, pstdev
from typing import TypedDict

from qdrant_client import QdrantClient

from src.detection.pipeline import ChunkScore, DetectionPipeline
from src.embeddings.embedder import Embedder
from src.retrieval.config import DEFAULT_CONFIG, PolicyConfig
from src.vectorstore.qdrant_client import get_client, sample_random, search
from src.vectorstore.schema import TrustTier


@dataclass(frozen=True)
class RawCandidate:
    """A single Qdrant search hit, before any policy decision is applied."""

    point_id: str
    chunk_id: str
    doc_id: str
    source_uri: str
    trust_tier: TrustTier
    text: str
    similarity_score: float


class GateDecision(StrEnum):
    ALLOW = "allow"
    DOWNWEIGHT = "downweight"
    DROP = "drop"


@dataclass(frozen=True)
class GatedCandidate:
    """A candidate after trust_tier/poison_score gating and frequency checks."""

    candidate: RawCandidate
    chunk_score: ChunkScore
    decision: GateDecision
    drop_reason: str | None
    policy_weight: float  # in [0, 1]; 0.0 iff decision is DROP
    adjusted_score: float  # candidate.similarity_score * policy_weight
    frequency_flagged: bool


@dataclass(frozen=True)
class RetrievalResult:
    """Final output of `retrieve()`: survivors only, ranked, plus audit counts."""

    query: str
    candidates: list[GatedCandidate]  # <= config.retrieval_top_k, sorted desc by adjusted_score
    fetched_count: int
    dropped_count: int
    frequency_flags: list[str]  # chunk_ids flagged among the final candidates


def _gate(
    trust_tier: TrustTier,
    poison_score: float,
    config: PolicyConfig = DEFAULT_CONFIG,
) -> tuple[GateDecision, float, str | None]:
    """Pure decision function: (trust_tier, poison_score) -> (decision, weight, drop_reason).

    QUARANTINED is a categorical drop regardless of poison_score, unless
    `config.allow_quarantined_tier` is set. Otherwise each tier has a
    (downweight_threshold, hard_drop_threshold) poison_score band: below the
    downweight threshold the candidate passes at full weight; between the two
    thresholds its weight ramps linearly down to `downweight_floor_multiplier`;
    at or above the hard-drop threshold it's dropped. Verified's band is
    strictly more permissive than unverified's -- the two signals compound,
    verified content isn't exempt from poison scoring.
    """
    if trust_tier == TrustTier.QUARANTINED and not config.allow_quarantined_tier:
        return GateDecision.DROP, 0.0, "quarantined_tier"

    bands = {
        TrustTier.VERIFIED: (
            config.poison_downweight_threshold_verified,
            config.poison_hard_drop_threshold_verified,
        ),
        TrustTier.UNVERIFIED: (
            config.poison_downweight_threshold_unverified,
            config.poison_hard_drop_threshold_unverified,
        ),
        # Only reachable when allow_quarantined_tier=True; unverified's band
        # is the closest analog for a quarantined chunk let through.
        TrustTier.QUARANTINED: (
            config.poison_downweight_threshold_unverified,
            config.poison_hard_drop_threshold_unverified,
        ),
    }
    downweight_threshold, hard_drop_threshold = bands[trust_tier]

    if poison_score >= hard_drop_threshold:
        return GateDecision.DROP, 0.0, "poison_score_hard_drop"
    if poison_score >= downweight_threshold:
        span = hard_drop_threshold - downweight_threshold
        ramp = (poison_score - downweight_threshold) / span if span > 0 else 1.0
        weight = 1.0 - ramp * (1.0 - config.downweight_floor_multiplier)
        return GateDecision.DOWNWEIGHT, weight, None
    return GateDecision.ALLOW, 1.0, None


@dataclass
class FrequencyTracker:
    """In-memory, process-lifetime retrieval-frequency counter.

    Flags a chunk_id whose cumulative retrieval count is a config-driven
    number of standard deviations above the mean count across every chunk_id
    observed so far -- a chunk retrieved far more often than its peers is a
    poisoning tell (a broad semantic-attractor attack surfacing across many
    unrelated queries). In-memory rather than persisted: no durable store
    exists anywhere in this repo yet, and Phase 7's observability layer is
    the natural home for that later. Known limitation: multiple concurrent
    server workers (Phase 5+ FastAPI) would each keep independent counts.

    Only non-DROPped candidates are recorded -- a poison-hard-dropped chunk
    is already gone, so tracking it would be redundant with the poison gate;
    this signal targets chunks that repeatedly *survive* that gate instead.
    """

    counts: dict[str, int] = field(default_factory=dict)
    total_observations: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, chunk_id: str) -> None:
        with self._lock:
            self.counts[chunk_id] = self.counts.get(chunk_id, 0) + 1
            self.total_observations += 1

    def is_anomalous(self, chunk_id: str, config: PolicyConfig = DEFAULT_CONFIG) -> bool:
        with self._lock:
            if self.total_observations < config.frequency_min_samples or len(self.counts) < 2:
                return False
            values = list(self.counts.values())
            std = pstdev(values)
            if std == 0:
                return False
            z_score = (self.counts.get(chunk_id, 0) - fmean(values)) / std
            return z_score >= config.frequency_zscore_threshold


class RetrievalState(TypedDict):
    """Minimal LangGraph state for the retrieval node. Phase 5's generation
    node is the next consumer and can extend this as needed."""

    query: str
    result: RetrievalResult | None


# Lazy module-level singletons: Embedder/DetectionPipeline load real model
# weights (real cost), and FrequencyTracker must persist across calls to mean
# anything, so all three are cached here by default. Every one is also
# override-able via retrieve()'s keyword params so it stays unit-testable
# with fakes without touching these globals.
_embedder: Embedder | None = None
_qdrant_client: QdrantClient | None = None
_detection_pipeline: DetectionPipeline | None = None
_frequency_tracker = FrequencyTracker()
_reference_sample: list[str] | None = None


def _default_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _default_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_client()
    return _qdrant_client


def _default_detection_pipeline(embedder: Embedder) -> DetectionPipeline:
    global _detection_pipeline
    if _detection_pipeline is None:
        _detection_pipeline = DetectionPipeline(embedder=embedder)
    return _detection_pipeline


def _get_reference_sample(client: QdrantClient, config: PolicyConfig) -> list[str] | None:
    """Lazily cache a random corpus sample used to stabilize cluster-anomaly
    scoring (see module docstring). Only a non-empty result is cached -- an
    empty/unreachable collection retries on the next call instead of
    permanently caching a miss."""
    global _reference_sample
    if _reference_sample is None:
        points = sample_random(
            client, config.retrieval_collection_name, config.cluster_reference_sample_size
        )
        texts = [point.payload["text"] for point in points if point.payload]
        _reference_sample = texts or None
    return _reference_sample


def retrieve(
    query: str,
    *,
    config: PolicyConfig = DEFAULT_CONFIG,
    embedder: Embedder | None = None,
    qdrant_client_instance: QdrantClient | None = None,
    detection_pipeline: DetectionPipeline | None = None,
    tracker: FrequencyTracker | None = None,
) -> RetrievalResult:
    """Embed `query`, search Qdrant, live-score candidates for poison, gate on
    trust_tier + poison_score, flag retrieval-frequency anomalies, and return
    up to `config.retrieval_top_k` survivors ranked by adjusted score.

    No LangGraph dependency -- directly callable and testable on its own.
    `retrieve_node` below is the thin LangGraph adapter.
    """
    embedder = embedder or _default_embedder()
    client = qdrant_client_instance or _default_qdrant_client()
    pipeline = detection_pipeline or _default_detection_pipeline(embedder)
    tracker = tracker or _frequency_tracker

    fetch_k = max(
        config.retrieval_top_k, ceil(config.retrieval_top_k * config.retrieval_overfetch_factor)
    )
    allowed_tiers = (
        None if config.allow_quarantined_tier else [TrustTier.VERIFIED, TrustTier.UNVERIFIED]
    )

    query_vector = embedder.embed_one(query).tolist()
    points = search(client, query_vector, config.retrieval_collection_name, fetch_k, allowed_tiers)

    raw_candidates = [
        RawCandidate(
            point_id=str(point.id),
            chunk_id=point.payload["chunk_id"],
            doc_id=point.payload["doc_id"],
            source_uri=point.payload["source_uri"],
            trust_tier=TrustTier(point.payload["trust_tier"]),
            text=point.payload["text"],
            similarity_score=point.score,
        )
        for point in points
    ]
    if not raw_candidates:
        return RetrievalResult(
            query=query, candidates=[], fetched_count=0, dropped_count=0, frequency_flags=[]
        )

    reference_texts = _get_reference_sample(client, config)
    chunk_scores = pipeline.score_chunks(
        [candidate.text for candidate in raw_candidates],
        [candidate.chunk_id for candidate in raw_candidates],
        reference_texts,
    )

    gated: list[GatedCandidate] = []
    for candidate, chunk_score in zip(raw_candidates, chunk_scores, strict=True):
        decision, weight, drop_reason = _gate(
            candidate.trust_tier, chunk_score.poison_score, config
        )

        frequency_flagged = False
        if decision != GateDecision.DROP:
            tracker.record(candidate.chunk_id)
            frequency_flagged = tracker.is_anomalous(candidate.chunk_id, config)
            if frequency_flagged and config.frequency_hard_block:
                decision = GateDecision.DROP
                weight = 0.0
                drop_reason = "frequency_anomaly_hard_block"

        gated.append(
            GatedCandidate(
                candidate=candidate,
                chunk_score=chunk_score,
                decision=decision,
                drop_reason=drop_reason,
                policy_weight=weight,
                adjusted_score=candidate.similarity_score * weight,
                frequency_flagged=frequency_flagged,
            )
        )

    not_dropped = [gc for gc in gated if gc.decision != GateDecision.DROP]
    survivors = sorted(not_dropped, key=lambda gc: gc.adjusted_score, reverse=True)
    final_candidates = survivors[: config.retrieval_top_k]

    return RetrievalResult(
        query=query,
        candidates=final_candidates,
        fetched_count=len(raw_candidates),
        dropped_count=len(gated) - len(survivors),
        frequency_flags=[
            gated_candidate.candidate.chunk_id
            for gated_candidate in final_candidates
            if gated_candidate.frequency_flagged
        ],
    )


def retrieve_node(state: RetrievalState) -> dict:
    """Thin LangGraph adapter around `retrieve`.

    Usable as `graph.add_node("retrieve", retrieve_node)`; returns a partial
    state update per LangGraph node convention.
    """
    return {"result": retrieve(state["query"])}
