"""Phase 8 multi-agent LangGraph: planner -> parallel retriever agents ->
synthesizer.

Uses LangGraph's `Send` API (`langgraph.types.Send`) for genuine parallel
fan-out: `plan_node` decomposes the query into sub-queries, a conditional
edge dispatches one `Send` per sub-query to `retrieve_agent_node` (LangGraph
runs these as parallel branches within one Pregel superstep), and
`synthesize_node` -- reachable only via a static edge from
`retrieve_agent_node` -- runs once every branch has completed, merges their
results, and calls Phase 5's `generate()` over the merged context.

`retrieve_agent_node` calls Phase 4's `retrieve()` completely unmodified --
same `DetectionPipeline` gating, no bypass -- so the Phase 4 ASR-reduction
guarantee is structurally preserved regardless of how many sub-queries a
question decomposes into. `merge_sub_results` returns a plain, unmodified
`RetrievalResult`, which is exactly what lets `generate()` and every existing
eval helper (`eval/run_eval.py`'s `_attack_succeeded`/`_precision_at_k`)
consume it with zero code changes.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.agentic.planner import plan_subqueries
from src.generation.generate import GenerationResult, generate
from src.retrieval.config import DEFAULT_CONFIG as DEFAULT_POLICY_CONFIG
from src.retrieval.config import PolicyConfig
from src.retrieval.policy_engine import GatedCandidate, RetrievalResult, retrieve


class RetrieverAgentInput(TypedDict):
    """Per-branch input for one Send-dispatched retriever agent. Deliberately
    NOT a subset of AgenticState -- a Send's arg dict only needs to satisfy
    the target node's own local shape, confirmed against the installed
    langgraph's own `Send` docstring example."""

    query: str


class AgenticState(TypedDict):
    """query/result/generation deliberately match GenerationState's exact
    field names (src/generation/generate.py) -- the same "compose across
    phases" convention RetrievalState -> GenerationState already
    established. sub_queries/sub_results are new to this phase.

    `sub_results` uses an `operator.add`-reducer so each parallel
    `retrieve_agent_node` branch's single-item list gets concatenated into
    one merged list, rather than one branch's write clobbering another's.
    """

    query: str
    sub_queries: list[str]
    sub_results: Annotated[list[RetrievalResult], operator.add]
    result: RetrievalResult | None
    generation: GenerationResult | None


def plan_node(state: AgenticState) -> dict:
    return {"sub_queries": plan_subqueries(state["query"])}


def route_to_retriever_agents(state: AgenticState) -> list[Send]:
    return [Send("retrieve_agent", {"query": sub_query}) for sub_query in state["sub_queries"]]


def retrieve_agent_node(state: RetrieverAgentInput) -> dict:
    """One parallel retriever-agent branch. Calls Phase 4's `retrieve()`
    completely unmodified -- no bypass of trust-tier/poison-score gating."""
    return {"sub_results": [retrieve(state["query"])]}


def merge_sub_results(
    query: str,
    sub_results: list[RetrievalResult],
    config: PolicyConfig = DEFAULT_POLICY_CONFIG,
) -> RetrievalResult:
    """Merge N independently-gated `RetrievalResult`s (one per planner
    sub-query, each already scored/gated by the unchanged Phase 4 policy
    engine) into one ranked, chunk_id-deduplicated `RetrievalResult` capped
    at `config.retrieval_top_k` -- the exact shape `generate()` already
    consumes, unmodified.

    Dedup keeps the max-`adjusted_score` occurrence of a repeated chunk_id
    (surfacing under multiple sub-queries is a stronger signal, but summing
    scores would move it off the single-`retrieve()`-call scale everything
    else is compared against). `fetched_count`/`dropped_count` sum across
    sub-results (an audit total of everything Phase 4's gate considered this
    call); the cross-sub-query re-rank/truncation below is not counted in
    `dropped_count` -- exactly like `retrieve()`'s own top_k truncation,
    `dropped_count` means "failed policy gating," not "didn't make the final
    ranked list."
    """
    best_by_chunk_id: dict[str, GatedCandidate] = {}
    for sub_result in sub_results:
        for gated in sub_result.candidates:
            existing = best_by_chunk_id.get(gated.candidate.chunk_id)
            if existing is None or gated.adjusted_score > existing.adjusted_score:
                best_by_chunk_id[gated.candidate.chunk_id] = gated

    ranked = sorted(best_by_chunk_id.values(), key=lambda gc: gc.adjusted_score, reverse=True)
    final_candidates = ranked[: config.retrieval_top_k]

    return RetrievalResult(
        query=query,
        candidates=final_candidates,
        fetched_count=sum(sr.fetched_count for sr in sub_results),
        dropped_count=sum(sr.dropped_count for sr in sub_results),
        frequency_flags=[gc.candidate.chunk_id for gc in final_candidates if gc.frequency_flagged],
    )


def synthesize_node(state: AgenticState) -> dict:
    merged = merge_sub_results(state["query"], state["sub_results"])
    return {"result": merged, "generation": generate(state["query"], merged)}


def build_agentic_graph() -> CompiledStateGraph:
    """Compile the planner -> parallel retriever agents -> synthesizer graph."""
    graph = StateGraph(AgenticState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve_agent", retrieve_agent_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_to_retriever_agents)
    graph.add_edge("retrieve_agent", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
