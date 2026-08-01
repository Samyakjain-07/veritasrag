import numpy as np
import pytest
from src.detection.config import DetectionConfig
from src.detection.perplexity import PerplexityScorer, normalize_perplexity


@pytest.fixture(scope="module")
def scorer():
    return PerplexityScorer("distilgpt2")


def test_compute_returns_positive_float(scorer):
    perplexity = scorer.compute("The sky is blue during a clear day.")

    assert isinstance(perplexity, float)
    assert perplexity > 0


def test_compute_batch_matches_compute(scorer):
    texts = ["hello world", "the sky is blue"]

    batch_scores = scorer.compute_batch(texts)
    individual_scores = [scorer.compute(t) for t in texts]

    assert np.allclose(batch_scores, individual_scores)


def test_gibberish_scores_higher_than_natural_text(scorer):
    natural = "The city council approved the new budget after a long public debate."
    gibberish = "purple elephant quickly maybe seventeen because ocean umbrella"

    assert scorer.compute(gibberish) > scorer.compute(natural)


def test_normalize_perplexity_clips_to_unit_range():
    config = DetectionConfig()
    raw_scores = [config.perplexity_low_anchor - 50, config.perplexity_high_anchor + 50]

    normalized = normalize_perplexity(raw_scores, config)

    assert normalized[0] == 0.0
    assert normalized[1] == 1.0


def test_normalize_perplexity_is_batch_independent():
    config = DetectionConfig()

    solo = normalize_perplexity([70.0], config)
    with_others = normalize_perplexity([10.0, 70.0, 200.0], config)

    assert solo[0] == with_others[1]
