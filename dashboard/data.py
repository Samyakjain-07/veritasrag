"""Pure data-loading/transform functions backing the Phase 7 Streamlit
dashboard (`dashboard/app.py`).

Deliberately has no `streamlit`/`opentelemetry` imports -- keeps this module
importable and unit-testable (`tests/dashboard/test_data.py`) without a
Streamlit process or a live OTel SDK, mirroring the "pure logic + thin
adapter" split already used throughout this repo (e.g. `retrieve()`/
`retrieve_node` in `src/retrieval/policy_engine.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from src.ingestion.provenance import ProvenanceSigner
from src.unlearning.receipt import load_receipt, verify_receipt
from src.vectorstore.qdrant_client import get_client, trust_tier_counts
from src.vectorstore.schema import COLLECTION_NAME

_SPAN_COLUMNS = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "start_time",
    "end_time",
    "duration_ms",
    "status_code",
    "service_name",
    "attributes",
)

_DETECTION_RATE_COLUMNS = (
    "start_time",
    "fetched_count",
    "allow_count",
    "downweight_count",
    "drop_count",
    "detection_rate",
)

_RECEIPT_COLUMNS = (
    "receipt_id",
    "chunk_id",
    "deleted_at",
    "requester",
    "found",
    "membership_check_passed",
    "reconstruction_check_passed",
    "recomputed_valid",
)


def load_trace_events(path: str | Path) -> list[dict]:
    """Read the JSONL trace file into a list of span-record dicts.

    Skips malformed lines defensively rather than raising: the exporter's
    lock (`src/observability/otel_setup.py::JSONLFileSpanExporter`) only
    serializes writer threads against each other, not against a concurrent
    reader observing a torn final line while a write is in flight. Returns
    `[]` if the file doesn't exist yet (fresh install, no queries run).
    """
    path = Path(path)
    if not path.exists():
        return []

    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def spans_to_dataframe(events: list[dict]) -> pd.DataFrame:
    """Normalize raw span-record dicts into a DataFrame.

    `attributes` is kept as a raw dict-valued column (not flattened into
    top-level columns) -- retrieve-span attributes and generate-span
    attributes don't collide/require handling each other's absent keys.
    """
    if not events:
        return pd.DataFrame(columns=_SPAN_COLUMNS)

    rows = []
    for event in events:
        start_ns = event.get("start_time_ns")
        end_ns = event.get("end_time_ns")
        has_both = start_ns is not None and end_ns is not None
        duration_ms = (end_ns - start_ns) / 1e6 if has_both else None
        start_time = pd.to_datetime(start_ns, unit="ns") if start_ns is not None else pd.NaT
        end_time = pd.to_datetime(end_ns, unit="ns") if end_ns is not None else pd.NaT
        rows.append(
            {
                "trace_id": event.get("trace_id"),
                "span_id": event.get("span_id"),
                "parent_span_id": event.get("parent_span_id"),
                "name": event.get("name"),
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "status_code": event.get("status_code"),
                "service_name": event.get("service_name"),
                "attributes": event.get("attributes") or {},
            }
        )
    return (
        pd.DataFrame(rows, columns=list(_SPAN_COLUMNS))
        .sort_values("start_time")
        .reset_index(drop=True)
    )


def detection_rate_timeseries(spans_df: pd.DataFrame) -> pd.DataFrame:
    """Per-query detection-rate series from `veritasrag.retrieval.retrieve` spans.

    `detection_rate = (downweight_count + poison_score_hard_drop_count) /
    fetched_count`, guarded against a zero `fetched_count` (-> `NaN`, not a
    crash). One row per query, chronologically sorted -- deliberately not
    time-bucketed/resampled: a local dev tool with occasional manual queries
    would produce mostly-empty buckets, so per-query rows are simpler and
    still answer "detection rate over time."
    """
    if spans_df.empty:
        return pd.DataFrame(columns=_DETECTION_RATE_COLUMNS)

    retrieve_spans = spans_df[spans_df["name"] == "veritasrag.retrieval.retrieve"]
    if retrieve_spans.empty:
        return pd.DataFrame(columns=_DETECTION_RATE_COLUMNS)

    rows = []
    for _, row in retrieve_spans.iterrows():
        attrs = row["attributes"] or {}
        fetched = attrs.get("veritasrag.retrieval.fetched_count", 0)
        allow = attrs.get("veritasrag.retrieval.allow_count", 0)
        downweight = attrs.get("veritasrag.retrieval.downweight_count", 0)
        drop = attrs.get("veritasrag.retrieval.drop_count", 0)
        hard_drop = attrs.get("veritasrag.retrieval.drop_reason.poison_score_hard_drop_count", 0)
        detection_rate = (downweight + hard_drop) / fetched if fetched else float("nan")
        rows.append(
            {
                "start_time": row["start_time"],
                "fetched_count": fetched,
                "allow_count": allow,
                "downweight_count": downweight,
                "drop_count": drop,
                "detection_rate": detection_rate,
            }
        )
    return (
        pd.DataFrame(rows, columns=list(_DETECTION_RATE_COLUMNS))
        .sort_values("start_time")
        .reset_index(drop=True)
    )


def live_trust_tier_distribution(
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, int]:
    """Thin wrapper around `get_client()`/`trust_tier_counts()` for a live
    snapshot of the Qdrant corpus's trust-tier makeup."""
    client = client or get_client()
    return trust_tier_counts(client, collection_name)


def load_receipt_files(receipt_dir: str | Path = "receipts") -> list[dict]:
    """Load every deletion receipt JSON file under `receipt_dir`."""
    receipt_dir = Path(receipt_dir)
    if not receipt_dir.exists():
        return []
    return [load_receipt(path) for path in sorted(receipt_dir.glob("*.json"))]


def verified_receipts(
    receipt_dir: str | Path = "receipts",
    signer: ProvenanceSigner | None = None,
) -> pd.DataFrame:
    """Deletion audit log: every receipt, with validity *recomputed live*
    via `verify_receipt()` rather than trusting any stored field -- there
    isn't one, and this matches the unlearning module's own verifiability
    emphasis (a receipt's whole point is that its validity is checkable, not
    just asserted).
    """
    signer = signer or ProvenanceSigner()
    receipts = load_receipt_files(receipt_dir)
    if not receipts:
        return pd.DataFrame(columns=_RECEIPT_COLUMNS)

    rows = [
        {
            "receipt_id": receipt.get("receipt_id"),
            "chunk_id": receipt.get("chunk_id"),
            "deleted_at": receipt.get("deleted_at"),
            "requester": receipt.get("requester"),
            "found": receipt.get("found"),
            "membership_check_passed": receipt.get("membership_check_passed"),
            "reconstruction_check_passed": receipt.get("reconstruction_check_passed"),
            "recomputed_valid": verify_receipt(receipt, signer),
        }
        for receipt in receipts
    ]
    return (
        pd.DataFrame(rows, columns=list(_RECEIPT_COLUMNS))
        .sort_values("deleted_at", ascending=False)
        .reset_index(drop=True)
    )
