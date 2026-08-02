import pytest
from src.retrieval.config import PolicyConfig
from src.retrieval.policy_engine import FrequencyTracker, GateDecision, _gate
from src.vectorstore.schema import TrustTier


def _gate_config(**overrides) -> PolicyConfig:
    return PolicyConfig(
        poison_downweight_threshold_verified=0.6,
        poison_hard_drop_threshold_verified=0.9,
        poison_downweight_threshold_unverified=0.3,
        poison_hard_drop_threshold_unverified=0.6,
        downweight_floor_multiplier=0.25,
        **overrides,
    )


def test_quarantined_tier_is_always_dropped_by_default():
    config = _gate_config()

    decision, weight, reason = _gate(TrustTier.QUARANTINED, 0.0, config)

    assert decision == GateDecision.DROP
    assert weight == 0.0
    assert reason == "quarantined_tier"


def test_quarantined_tier_allowed_through_uses_unverified_band():
    config = _gate_config(allow_quarantined_tier=True)

    below_band = _gate(TrustTier.QUARANTINED, 0.1, config)
    above_band = _gate(TrustTier.QUARANTINED, 0.7, config)

    assert below_band[0] == GateDecision.ALLOW
    assert above_band[0] == GateDecision.DROP


@pytest.mark.parametrize("tier", [TrustTier.VERIFIED, TrustTier.UNVERIFIED])
def test_low_poison_score_allows_at_full_weight(tier):
    config = _gate_config()

    decision, weight, reason = _gate(tier, 0.0, config)

    assert decision == GateDecision.ALLOW
    assert weight == 1.0
    assert reason is None


def test_verified_hard_drop_threshold():
    config = _gate_config()

    decision, weight, reason = _gate(TrustTier.VERIFIED, 0.9, config)

    assert decision == GateDecision.DROP
    assert weight == 0.0
    assert reason == "poison_score_hard_drop"


def test_unverified_hard_drop_threshold():
    config = _gate_config()

    decision, weight, reason = _gate(TrustTier.UNVERIFIED, 0.6, config)

    assert decision == GateDecision.DROP
    assert reason == "poison_score_hard_drop"


def test_verified_downweight_ramp_midpoint():
    config = _gate_config()  # verified band: (0.6, 0.9), floor=0.25

    decision, weight, reason = _gate(TrustTier.VERIFIED, 0.75, config)

    assert decision == GateDecision.DOWNWEIGHT
    assert weight == pytest.approx(0.625)  # 1.0 - 0.5 * (1 - 0.25)
    assert reason is None


def test_unverified_downweight_ramp_midpoint():
    config = _gate_config()  # unverified band: (0.3, 0.6), floor=0.25

    decision, weight, reason = _gate(TrustTier.UNVERIFIED, 0.45, config)

    assert decision == GateDecision.DOWNWEIGHT
    assert weight == pytest.approx(0.625)  # 1.0 - 0.5 * (1 - 0.25)


def test_downweight_threshold_boundary_is_full_weight():
    """At the exact downweight threshold, the ramp fraction is 0 -> weight 1.0
    even though the decision has already transitioned to DOWNWEIGHT."""
    config = _gate_config()

    decision, weight, _ = _gate(TrustTier.VERIFIED, 0.6, config)

    assert decision == GateDecision.DOWNWEIGHT
    assert weight == pytest.approx(1.0)


def test_verified_band_is_strictly_more_permissive_than_unverified():
    config = _gate_config()
    poison_score = 0.4  # inside unverified's downweight band, below verified's

    verified_decision, _, _ = _gate(TrustTier.VERIFIED, poison_score, config)
    unverified_decision, _, _ = _gate(TrustTier.UNVERIFIED, poison_score, config)

    assert verified_decision == GateDecision.ALLOW
    assert unverified_decision == GateDecision.DOWNWEIGHT


def test_frequency_tracker_records_counts():
    tracker = FrequencyTracker()

    tracker.record("chunk-a")
    tracker.record("chunk-a")
    tracker.record("chunk-b")

    assert tracker.counts == {"chunk-a": 2, "chunk-b": 1}
    assert tracker.total_observations == 3


def test_frequency_tracker_not_anomalous_before_min_samples():
    config = PolicyConfig(frequency_min_samples=100, frequency_zscore_threshold=0.01)
    tracker = FrequencyTracker()
    for _ in range(5):
        tracker.record("spike")

    assert tracker.is_anomalous("spike", config) is False


def test_frequency_tracker_not_anomalous_with_single_distinct_chunk():
    config = PolicyConfig(frequency_min_samples=1, frequency_zscore_threshold=0.01)
    tracker = FrequencyTracker()
    for _ in range(10):
        tracker.record("only-chunk")

    assert tracker.is_anomalous("only-chunk", config) is False


def test_frequency_tracker_not_anomalous_when_counts_uniform():
    config = PolicyConfig(frequency_min_samples=1, frequency_zscore_threshold=0.01)
    tracker = FrequencyTracker()
    for chunk_id in ("a", "b", "c"):
        tracker.record(chunk_id)

    assert tracker.is_anomalous("a", config) is False


def test_frequency_tracker_flags_statistical_outlier():
    # counts: a=1, b=1, c=1, d=1, spike=6 -> mean=2.0, pstdev=2.0, z(spike)=2.0
    config = PolicyConfig(frequency_min_samples=5, frequency_zscore_threshold=2.0)
    tracker = FrequencyTracker()
    for chunk_id in ("a", "b", "c", "d"):
        tracker.record(chunk_id)
    for _ in range(6):
        tracker.record("spike")

    assert tracker.is_anomalous("spike", config) is True
    assert tracker.is_anomalous("a", config) is False


def test_frequency_tracker_unseen_chunk_id_does_not_raise():
    config = PolicyConfig(frequency_min_samples=1, frequency_zscore_threshold=2.0)
    tracker = FrequencyTracker()
    for chunk_id in ("a", "b"):
        tracker.record(chunk_id)

    assert tracker.is_anomalous("never-seen", config) is False
