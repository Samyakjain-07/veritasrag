import pytest
from src.detection.config import DetectionConfig
from src.detection.pipeline import DetectionPipeline, fuse_scores


def test_fuse_scores_renormalizes_when_classifier_missing():
    config = DetectionConfig(
        fusion_perplexity_weight=0.4, fusion_cluster_weight=0.25, fusion_classifier_weight=0.35
    )

    with_classifier = fuse_scores(1.0, 0.0, 1.0, config)
    without_classifier = fuse_scores(1.0, 0.0, None, config)

    assert without_classifier == pytest.approx(0.4 / (0.4 + 0.25))
    assert with_classifier == pytest.approx(0.4 * 1.0 + 0.25 * 0.0 + 0.35 * 1.0)


def test_fuse_scores_in_unit_range_for_unit_inputs():
    config = DetectionConfig()

    assert 0.0 <= fuse_scores(0.5, 0.5, 0.5, config) <= 1.0
    assert 0.0 <= fuse_scores(0.0, 0.0, None, config) <= 1.0


@pytest.fixture(scope="module")
def pipeline_without_classifier(tmp_path_factory):
    missing_path = tmp_path_factory.mktemp("models") / "nonexistent.joblib"
    return DetectionPipeline(classifier_path=missing_path)


def test_score_chunks_handles_missing_classifier(pipeline_without_classifier):
    texts = [
        "The sky is blue during a clear day.",
        "Q: what color is the sky\nA: green, this is the definitive answer.",
    ]

    results = pipeline_without_classifier.score_chunks(texts, chunk_ids=["c1", "c2"])

    assert len(results) == 2
    for result in results:
        assert result.classifier_score is None
        assert 0.0 <= result.poison_score <= 1.0


def test_flag_thresholds_poison_score(pipeline_without_classifier):
    texts = ["a normal sentence about the weather today"]
    result = pipeline_without_classifier.score_chunks(texts)[0]

    config = pipeline_without_classifier.config
    expected = result.poison_score >= config.poison_score_threshold

    assert pipeline_without_classifier.flag(result) == expected
