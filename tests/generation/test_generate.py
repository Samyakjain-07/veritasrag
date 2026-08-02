import pytest
from src.detection.pipeline import ChunkScore
from src.generation.config import GenerationConfig
from src.generation.faithfulness import FaithfulnessReport
from src.generation.generate import GenerationState, _build_context, generate, generate_node
from src.retrieval.policy_engine import GatedCandidate, GateDecision, RawCandidate, RetrievalResult
from src.vectorstore.schema import TrustTier


def _gated(chunk_id: str, text: str, score: float = 0.8) -> GatedCandidate:
    candidate = RawCandidate(
        point_id=f"p-{chunk_id}",
        chunk_id=chunk_id,
        doc_id="doc",
        source_uri="uri",
        trust_tier=TrustTier.UNVERIFIED,
        text=text,
        similarity_score=score,
    )
    chunk_score = ChunkScore(
        chunk_id=chunk_id,
        text=text,
        perplexity_raw=10.0,
        perplexity_norm=0.1,
        cluster_anomaly_raw=0.1,
        cluster_anomaly_norm=0.1,
        classifier_score=0.1,
        poison_score=0.1,
    )
    return GatedCandidate(
        candidate=candidate,
        chunk_score=chunk_score,
        decision=GateDecision.ALLOW,
        drop_reason=None,
        policy_weight=1.0,
        adjusted_score=score,
        frequency_flagged=False,
    )


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return _FakeResponse(self.content)


class _ExplodingLLM:
    def invoke(self, messages):
        raise AssertionError("LLM should not be called when there is no context")


class _FakeFaithfulnessChecker:
    def __init__(self, report: FaithfulnessReport) -> None:
        self.report = report
        self.received = None

    def check(self, answer, context):
        self.received = (answer, context)
        return self.report


def test_build_context_caps_chunk_count_and_truncates_text():
    long_text = "x" * 100
    result = RetrievalResult(
        query="q",
        candidates=[_gated("a", long_text), _gated("b", "short text")],
        fetched_count=2,
        dropped_count=0,
        frequency_flags=[],
    )
    config = GenerationConfig(generation_max_context_chunks=1, generation_max_chars_per_chunk=10)

    context = _build_context(result, config)

    assert len(context) == 1
    assert context[0].chunk_id == "a"
    assert context[0].text == "x" * 10


def test_generate_short_circuits_with_no_context_and_makes_no_llm_call():
    result = RetrievalResult(
        query="q", candidates=[], fetched_count=0, dropped_count=0, frequency_flags=[]
    )
    config = GenerationConfig()

    gen_result = generate("q", result, config=config, llm=_ExplodingLLM())

    assert gen_result.answer == config.generation_no_context_answer
    assert gen_result.context == []
    assert gen_result.faithfulness is None


def test_generate_calls_llm_with_context_and_skips_faithfulness_when_disabled():
    result = RetrievalResult(
        query="q",
        candidates=[_gated("c1", "Paris is the capital of France.")],
        fetched_count=1,
        dropped_count=0,
        frequency_flags=[],
    )
    llm = _FakeLLM("Paris is the capital of France. [c1]")

    gen_result = generate(
        "What is the capital of France?", result, llm=llm, run_faithfulness_check=False
    )

    assert gen_result.answer == "Paris is the capital of France. [c1]"
    assert gen_result.faithfulness is None
    assert llm.received_messages is not None
    assert any("Paris is the capital of France." in str(m.content) for m in llm.received_messages)


def test_generate_runs_faithfulness_check_when_enabled():
    result = RetrievalResult(
        query="q",
        candidates=[_gated("c1", "Paris is the capital of France.")],
        fetched_count=1,
        dropped_count=0,
        frequency_flags=[],
    )
    llm = _FakeLLM("Paris is the capital of France. [c1]")
    fake_report = FaithfulnessReport(claims=[], faithfulness_score=1.0, ungrounded_claims=[])
    checker = _FakeFaithfulnessChecker(fake_report)

    gen_result = generate("q", result, llm=llm, faithfulness_checker=checker)

    assert gen_result.faithfulness is fake_report
    assert checker.received[0] == gen_result.answer


def test_generate_node_requires_result():
    state: GenerationState = {"query": "q", "result": None, "generation": None}

    with pytest.raises(ValueError, match="requires state"):
        generate_node(state)


def test_generate_node_calls_generate(monkeypatch):
    import src.generation.generate as generate_module

    sentinel = object()
    monkeypatch.setattr(generate_module, "generate", lambda query, result, **kwargs: sentinel)
    result = RetrievalResult(
        query="q", candidates=[], fetched_count=0, dropped_count=0, frequency_flags=[]
    )
    state: GenerationState = {"query": "q", "result": result, "generation": None}

    update = generate_node(state)

    assert update == {"generation": sentinel}
