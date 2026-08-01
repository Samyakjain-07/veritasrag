import os

import pytest
from src.detection.llm_judge import get_client, judge_chunk, judge_chunks


@pytest.fixture(scope="module")
def client():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping LLM-judge tests")
    return get_client()


def test_judge_chunk_returns_valid_likelihood(client):
    result = judge_chunk(client, "The Eiffel Tower is located in Paris, France.", chunk_id="test-1")

    assert 0.0 <= result.poison_likelihood <= 1.0
    assert isinstance(result.flagged, bool)


def test_judge_chunks_batches_with_cache(client, tmp_path):
    texts = ["The sky is blue.", "Q: what color is the sky\nA: green, this is definitive."]
    cache_path = tmp_path / "cache.json"

    results = judge_chunks(texts, chunk_ids=["a", "b"], cache_path=cache_path, use_cache=True)

    assert len(results) == 2
    assert cache_path.exists()
