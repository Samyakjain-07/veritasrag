import pytest
from src.generation.config import FaithfulnessConfig
from src.generation.faithfulness import ContextChunk, FaithfulnessChecker, split_claims


def test_split_claims_splits_on_sentence_boundaries():
    claims = split_claims(
        "Paris is the capital of France. It has a population of over two million."
    )

    assert claims == [
        "Paris is the capital of France.",
        "It has a population of over two million.",
    ]


def test_split_claims_drops_trailing_citation_only_fragment():
    claims = split_claims("Dadabhai Naoroji wrote the famous work. [clean-07331-0]")

    assert claims == ["Dadabhai Naoroji wrote the famous work."]


def test_split_claims_keeps_inline_citation_within_a_claim():
    claims = split_claims("Dadabhai Naoroji wrote the famous work [clean-07331-0].")

    assert claims == ["Dadabhai Naoroji wrote the famous work [clean-07331-0]."]


def test_split_claims_drops_short_fragments():
    config = FaithfulnessConfig(faithfulness_min_claim_chars=20)

    claims = split_claims("Yes. Paris is the capital of France.", config)

    assert claims == ["Paris is the capital of France."]


@pytest.fixture(scope="module")
def checker():
    return FaithfulnessChecker(
        FaithfulnessConfig(faithfulness_nli_model_name="cross-encoder/nli-deberta-v3-xsmall")
    )


def test_check_grounds_claim_in_supporting_chunk(checker):
    context = [ContextChunk(chunk_id="c1", text="The Eiffel Tower is located in Paris, France.")]

    report = checker.check("The Eiffel Tower is in Paris.", context)

    assert len(report.claims) == 1
    assert report.claims[0].grounded is True
    assert report.claims[0].best_chunk_id == "c1"
    assert report.faithfulness_score == 1.0
    assert report.ungrounded_claims == []


def test_check_flags_unsupported_claim_as_ungrounded(checker):
    context = [ContextChunk(chunk_id="c1", text="The Eiffel Tower is located in Paris, France.")]

    report = checker.check("The moon is made of green cheese.", context)

    assert report.claims[0].grounded is False
    assert report.faithfulness_score == 0.0
    assert report.ungrounded_claims == ["The moon is made of green cheese."]


def test_check_picks_best_supporting_chunk_among_several(checker):
    context = [
        ContextChunk(chunk_id="irrelevant", text="Bananas are a good source of potassium."),
        ContextChunk(chunk_id="supporting", text="The Eiffel Tower is located in Paris, France."),
    ]

    report = checker.check("The Eiffel Tower is in Paris.", context)

    assert report.claims[0].best_chunk_id == "supporting"


def test_check_no_claims_is_vacuously_faithful(checker):
    report = checker.check("", [ContextChunk(chunk_id="c1", text="Some context.")])

    assert report.claims == []
    assert report.faithfulness_score == 1.0
    assert report.ungrounded_claims == []


def test_check_no_context_is_wholly_unfaithful(checker):
    report = checker.check("This is a claim.", [])

    assert report.claims[0].grounded is False
    assert report.claims[0].best_chunk_id is None
    assert report.faithfulness_score == 0.0
