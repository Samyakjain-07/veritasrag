import numpy as np
import pytest
from src.embeddings.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embed_returns_expected_shape(embedder):
    vectors = embedder.embed(["hello world", "a different sentence"])

    assert vectors.shape == (2, embedder.dimension)


def test_embed_is_deterministic(embedder):
    vectors_a = embedder.embed(["hello world"])
    vectors_b = embedder.embed(["hello world"])

    assert np.allclose(vectors_a, vectors_b)


def test_embed_one_matches_embed(embedder):
    single = embedder.embed_one("hello world")
    batch = embedder.embed(["hello world"])

    assert np.allclose(single, batch[0])
