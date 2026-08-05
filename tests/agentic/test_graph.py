import dataclasses

import src.agentic.graph as graph_module
from src.agentic.graph import build_agentic_graph, merge_sub_results
from src.detection.pipeline import ChunkScore
from src.generation.generate import GenerationResult
from src.retrieval.config import PolicyConfig
from src.retrieval.policy_engine import GatedCandidate, GateDecision, RawCandidate, RetrievalResult
from src.vectorstore.schema import TrustTier


def _gated(chunk_id: str, score: float) -> GatedCandidate:
    candidate = RawCandidate(
        point_id=f"p-{chunk_id}",
        chunk_id=chunk_id,
        doc_id="doc",
        source_uri="uri",
        trust_tier=TrustTier.UNVERIFIED,
        text=f"text for {chunk_id}",
        similarity_score=score,
    )
    chunk_score = ChunkScore(
        chunk_id=chunk_id,
        text=candidate.text,
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


def _result(candidates, fetched_count, dropped_count=0, frequency_flags=None) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        candidates=candidates,
        fetched_count=fetched_count,
        dropped_count=dropped_count,
        frequency_flags=frequency_flags or [],
    )


def test_merge_sub_results_dedupes_keeping_max_adjusted_score():
    sub_results = [
        _result([_gated("a", 0.5), _gated("b", 0.9)], fetched_count=2),
        _result([_gated("a", 0.8), _gated("c", 0.3)], fetched_count=2),
    ]

    merged = merge_sub_results("q", sub_results)

    by_id = {gc.candidate.chunk_id: gc for gc in merged.candidates}
    assert by_id["a"].adjusted_score == 0.8  # max across sub-queries kept
    assert set(by_id) == {"a", "b", "c"}
    assert merged.fetched_count == 4
    assert merged.dropped_count == 0


def test_merge_sub_results_ranks_desc_and_caps_at_top_k():
    sub_results = [_result([_gated("a", 0.1), _gated("b", 0.9), _gated("c", 0.5)], fetched_count=3)]
    config = PolicyConfig(retrieval_top_k=2)

    merged = merge_sub_results("q", sub_results, config=config)

    assert [gc.candidate.chunk_id for gc in merged.candidates] == ["b", "c"]


def test_merge_sub_results_recomputes_frequency_flags_from_final_list():
    flagged = dataclasses.replace(_gated("a", 0.9), frequency_flagged=True)
    sub_results = [_result([flagged, _gated("b", 0.1)], fetched_count=2)]

    merged = merge_sub_results("q", sub_results)

    assert merged.frequency_flags == ["a"]


def test_merge_sub_results_empty_input_returns_empty_result():
    merged = merge_sub_results("q", [])

    assert merged.candidates == []
    assert merged.fetched_count == 0
    assert merged.dropped_count == 0
    assert merged.frequency_flags == []


def test_build_agentic_graph_fans_out_merges_and_synthesizes(monkeypatch):
    """Smoke test of the compiled planner -> parallel retriever agents ->
    synthesizer graph shape, with plan_subqueries/retrieve/generate faked out
    so it needs no live LLM/Qdrant/model. Confirms: (1) the planner's
    sub-queries genuinely dispatch one retrieve() call each (parallel
    fan-out actually happened, not just a single pass-through), and (2) the
    synthesizer receives the merged, deduplicated result -- not just one
    branch's raw result.
    """
    monkeypatch.setattr(
        graph_module, "plan_subqueries", lambda query: ["sub-query-1", "sub-query-2"]
    )

    calls: list[str] = []

    def fake_retrieve(query):
        calls.append(query)
        if query == "sub-query-1":
            return _result([_gated("shared", 0.5), _gated("only-in-1", 0.9)], fetched_count=2)
        return _result([_gated("shared", 0.8), _gated("only-in-2", 0.3)], fetched_count=2)

    monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)

    captured_merged: list[RetrievalResult] = []

    def fake_generate(query, result):
        captured_merged.append(result)
        return GenerationResult(
            query=query, answer="synthesized answer", context=[], faithfulness=None
        )

    monkeypatch.setattr(graph_module, "generate", fake_generate)

    graph = build_agentic_graph()
    final_state = graph.invoke({"query": "original question", "sub_queries": [], "sub_results": []})

    assert sorted(calls) == ["sub-query-1", "sub-query-2"]
    assert final_state["generation"].answer == "synthesized answer"

    merged = final_state["result"]
    chunk_ids = {gc.candidate.chunk_id for gc in merged.candidates}
    assert chunk_ids == {"shared", "only-in-1", "only-in-2"}
    by_id = {gc.candidate.chunk_id: gc for gc in merged.candidates}
    assert by_id["shared"].adjusted_score == 0.8  # max across the two branches

    assert captured_merged[0] is merged
