import json
import math

import pytest
from src.ingestion.provenance import ProvenanceSigner
from src.unlearning.config import UnlearningConfig
from src.unlearning.receipt import build_receipt, save_receipt, sign_receipt

from dashboard import data

_SPAN_COLUMNS = [
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
]


def _make_event(
    name: str,
    start_ns: int,
    end_ns: int,
    *,
    attributes: dict | None = None,
    parent_span_id: str | None = None,
    trace_id: str = "a" * 32,
    span_id: str = "b" * 16,
) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time_ns": start_ns,
        "end_time_ns": end_ns,
        "status_code": "OK",
        "service_name": "veritasrag",
        "attributes": attributes or {},
    }


def test_load_trace_events_missing_file_returns_empty(tmp_path):
    assert data.load_trace_events(tmp_path / "nope.jsonl") == []


def test_load_trace_events_skips_malformed_lines(tmp_path):
    path = tmp_path / "traces.jsonl"
    good = _make_event("x", 1, 2)
    path.write_text(json.dumps(good) + "\n{not valid json\n", encoding="utf-8")

    events = data.load_trace_events(path)

    assert events == [good]


def test_spans_to_dataframe_empty():
    df = data.spans_to_dataframe([])

    assert list(df.columns) == _SPAN_COLUMNS
    assert df.empty


def test_spans_to_dataframe_computes_duration_and_sorts():
    events = [
        _make_event("second", 2_000_000_000, 2_500_000_000, span_id="s2"),
        _make_event("first", 1_000_000_000, 1_200_000_000, span_id="s1"),
    ]

    df = data.spans_to_dataframe(events)

    assert list(df["name"]) == ["first", "second"]
    assert df.iloc[0]["duration_ms"] == pytest.approx(200.0)
    assert df.iloc[1]["duration_ms"] == pytest.approx(500.0)


def test_spans_to_dataframe_keeps_attributes_as_dict():
    events = [_make_event("x", 1, 2, attributes={"foo": 1})]

    df = data.spans_to_dataframe(events)

    assert df.iloc[0]["attributes"] == {"foo": 1}


def test_detection_rate_timeseries_computes_rate():
    events = [
        _make_event(
            "veritasrag.retrieval.retrieve",
            1_000_000_000,
            1_100_000_000,
            attributes={
                "veritasrag.retrieval.fetched_count": 10,
                "veritasrag.retrieval.allow_count": 6,
                "veritasrag.retrieval.downweight_count": 2,
                "veritasrag.retrieval.drop_count": 2,
                "veritasrag.retrieval.drop_reason.poison_score_hard_drop_count": 2,
            },
        ),
        _make_event("veritasrag.generation.generate", 1_100_000_000, 1_200_000_000),
    ]

    rate_df = data.detection_rate_timeseries(data.spans_to_dataframe(events))

    assert len(rate_df) == 1
    assert rate_df.iloc[0]["fetched_count"] == 10
    assert rate_df.iloc[0]["detection_rate"] == pytest.approx((2 + 2) / 10)


def test_detection_rate_timeseries_handles_zero_fetched_count():
    events = [
        _make_event(
            "veritasrag.retrieval.retrieve",
            1_000_000_000,
            1_100_000_000,
            attributes={"veritasrag.retrieval.fetched_count": 0},
        )
    ]

    rate_df = data.detection_rate_timeseries(data.spans_to_dataframe(events))

    assert math.isnan(rate_df.iloc[0]["detection_rate"])


def test_detection_rate_timeseries_empty_when_no_retrieve_spans():
    events = [_make_event("veritasrag.generation.generate", 1, 2)]

    rate_df = data.detection_rate_timeseries(data.spans_to_dataframe(events))

    assert rate_df.empty


def test_detection_rate_timeseries_empty_dataframe_input():
    rate_df = data.detection_rate_timeseries(data.spans_to_dataframe([]))

    assert rate_df.empty


def test_live_trust_tier_distribution_thin_wrapper(monkeypatch):
    calls = {}

    def fake_counts(client, collection_name):
        calls["client"] = client
        calls["collection_name"] = collection_name
        return {"verified": 3, "unverified": 5}

    monkeypatch.setattr(data, "trust_tier_counts", fake_counts)
    fake_client = object()

    result = data.live_trust_tier_distribution(client=fake_client, collection_name="custom")

    assert result == {"verified": 3, "unverified": 5}
    assert calls == {"client": fake_client, "collection_name": "custom"}


def test_load_receipt_files_missing_dir_returns_empty(tmp_path):
    assert data.load_receipt_files(tmp_path / "nope") == []


def test_verified_receipts_recomputes_validity(tmp_path):
    signer = ProvenanceSigner(tmp_path / "keys")
    receipt_dir = tmp_path / "receipts"
    body = build_receipt(
        chunk_id="clean-00001-0",
        point_id="p1",
        collection_name="veritasrag_chunks",
        content_hash="deadbeef",
        found=True,
        requester="tester",
    )
    receipt = sign_receipt(body, signer)
    save_receipt(receipt, config=UnlearningConfig(receipt_dir=str(receipt_dir)))

    df = data.verified_receipts(str(receipt_dir), signer=signer)

    assert len(df) == 1
    assert df.iloc[0]["chunk_id"] == "clean-00001-0"
    assert bool(df.iloc[0]["recomputed_valid"]) is True

    tampered_path = next(receipt_dir.glob("*.json"))
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["requester"] = "attacker"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

    df_after_tamper = data.verified_receipts(str(receipt_dir), signer=signer)

    assert bool(df_after_tamper.iloc[0]["recomputed_valid"]) is False


def test_verified_receipts_empty_dir_returns_empty(tmp_path):
    df = data.verified_receipts(str(tmp_path / "receipts"))

    assert df.empty
