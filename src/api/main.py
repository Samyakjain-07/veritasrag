"""Phase 5 FastAPI service: wires the Phase 4 retrieval policy engine and the
Phase 5 generation node into a single LangGraph `StateGraph` and exposes it as
`POST /query`.

`build_graph()` is the whole end-to-end loop: `retrieve_node` (trust-tier +
poison-score gating, frequency-anomaly flagging) feeds its `RetrievalResult`
straight into `generate_node` (LLM call + NLI faithfulness check) via the
shared `GenerationState` schema -- see `src/generation/generate.py`'s
docstring for why the two nodes' state fields line up.
"""

from __future__ import annotations

from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from src.generation.generate import GenerationResult, GenerationState, generate_node
from src.observability.otel_setup import get_tracer
from src.retrieval.policy_engine import GatedCandidate, RetrievalResult, retrieve_node


def build_graph() -> CompiledStateGraph:
    """Compile the retrieve -> generate LangGraph used by the `/query` endpoint."""
    graph = StateGraph(GenerationState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


app = FastAPI(
    title="VeritasRAG",
    description="Trustworthy RAG integrity layer: poisoning-defended retrieval, "
    "faithfulness-checked generation.",
)

_graph: CompiledStateGraph | None = None


def _get_graph() -> CompiledStateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class QueryRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    chunk_id: str
    source_uri: str
    trust_tier: str
    decision: str
    similarity_score: float
    adjusted_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse]
    faithfulness_score: float | None
    ungrounded_claims: list[str]
    fetched_count: int
    dropped_count: int
    frequency_flags: list[str]


def _citations_from_result(result: RetrievalResult) -> list[CitationResponse]:
    return [_citation_from_candidate(gated) for gated in result.candidates]


def _citation_from_candidate(gated: GatedCandidate) -> CitationResponse:
    return CitationResponse(
        chunk_id=gated.candidate.chunk_id,
        source_uri=gated.candidate.source_uri,
        trust_tier=gated.candidate.trust_tier.value,
        decision=gated.decision.value,
        similarity_score=gated.candidate.similarity_score,
        adjusted_score=gated.adjusted_score,
    )


def _response_from_state(
    question: str, result: RetrievalResult, generation: GenerationResult
) -> QueryResponse:
    faithfulness = generation.faithfulness
    return QueryResponse(
        question=question,
        answer=generation.answer,
        citations=_citations_from_result(result),
        faithfulness_score=faithfulness.faithfulness_score if faithfulness else None,
        ungrounded_claims=faithfulness.ungrounded_claims if faithfulness else [],
        fetched_count=result.fetched_count,
        dropped_count=result.dropped_count,
        frequency_flags=result.frequency_flags,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the full retrieve -> generate -> faithfulness-check loop for one question."""
    with get_tracer(__name__).start_as_current_span("veritasrag.query") as span:
        span.set_attribute("veritasrag.query.question", request.question[:200])
        initial_state = {"query": request.question, "result": None, "generation": None}
        final_state = _get_graph().invoke(initial_state)
        result: RetrievalResult = final_state["result"]
        generation: GenerationResult = final_state["generation"]
        return _response_from_state(request.question, result, generation)
