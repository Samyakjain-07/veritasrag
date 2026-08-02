from fastapi.testclient import TestClient
from src.api import main as api_main
from src.detection.pipeline import ChunkScore
from src.generation.faithfulness import ClaimFaithfulness, ContextChunk, FaithfulnessReport
from src.generation.generate import GenerationResult
from src.retrieval.policy_engine import GatedCandidate, GateDecision, RawCandidate, RetrievalResult
from src.vectorstore.schema import TrustTier


def _fake_gated_candidate(chunk_id: str = "c1", score: float = 0.9) -> GatedCandidate:
    text = "Paris is the capital of France."
    candidate = RawCandidate(
        point_id="p1",
        chunk_id=chunk_id,
        doc_id="doc1",
        source_uri="uri1",
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


class _FakeGraph:
    """Stands in for the compiled LangGraph so the test never touches
    Qdrant, the embedder, or a real LLM/NLI model."""

    def invoke(self, state: dict) -> dict:
        gated = _fake_gated_candidate()
        result = RetrievalResult(
            query=state["query"],
            candidates=[gated],
            fetched_count=1,
            dropped_count=0,
            frequency_flags=[],
        )
        faithfulness = FaithfulnessReport(
            claims=[
                ClaimFaithfulness(
                    claim="Paris is the capital of France.",
                    best_chunk_id="c1",
                    entailment_score=0.95,
                    grounded=True,
                )
            ],
            faithfulness_score=1.0,
            ungrounded_claims=[],
        )
        generation = GenerationResult(
            query=state["query"],
            answer="Paris is the capital of France. [c1]",
            context=[ContextChunk(chunk_id="c1", text=gated.candidate.text)],
            faithfulness=faithfulness,
        )
        return {"query": state["query"], "result": result, "generation": generation}


def test_health_endpoint():
    client = TestClient(api_main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr(api_main, "_graph", _FakeGraph())
    client = TestClient(api_main.app)

    response = client.post("/query", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is the capital of France?"
    assert body["answer"] == "Paris is the capital of France. [c1]"
    assert body["citations"] == [
        {
            "chunk_id": "c1",
            "source_uri": "uri1",
            "trust_tier": "unverified",
            "decision": "allow",
            "similarity_score": 0.9,
            "adjusted_score": 0.9,
        }
    ]
    assert body["faithfulness_score"] == 1.0
    assert body["ungrounded_claims"] == []
    assert body["fetched_count"] == 1
    assert body["dropped_count"] == 0
    assert body["frequency_flags"] == []


def test_query_endpoint_handles_no_faithfulness_report(monkeypatch):
    class _NoContextGraph:
        def invoke(self, state: dict) -> dict:
            result = RetrievalResult(
                query=state["query"],
                candidates=[],
                fetched_count=0,
                dropped_count=0,
                frequency_flags=[],
            )
            generation = GenerationResult(
                query=state["query"],
                answer="I don't have enough verified information in the trusted context "
                "to answer that.",
                context=[],
                faithfulness=None,
            )
            return {"query": state["query"], "result": result, "generation": generation}

    monkeypatch.setattr(api_main, "_graph", _NoContextGraph())
    client = TestClient(api_main.app)

    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert body["faithfulness_score"] is None
    assert body["ungrounded_claims"] == []
