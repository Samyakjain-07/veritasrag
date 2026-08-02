"""Phase 5 generation node: builds a prompt from the Phase 4 policy engine's
gated context, calls the configured LLM provider through
`src/generation/llm_client.py::get_llm()` (never a provider SDK directly, per
CLAUDE.md), and runs a post-hoc NLI faithfulness check on the answer.

`generate()` carries no LangGraph dependency and is directly callable/testable
on its own -- same split as `src/retrieval/policy_engine.py::retrieve` /
`retrieve_node`. `generate_node` is the thin LangGraph adapter, and
`GenerationState` extends `RetrievalState`'s exact `query`/`result` fields so
`retrieve_node` and `generate_node` compose in one graph (see
`src/api/main.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.generation.config import (
    DEFAULT_FAITHFULNESS_CONFIG,
    DEFAULT_GENERATION_CONFIG,
    FaithfulnessConfig,
    GenerationConfig,
)
from src.generation.faithfulness import ContextChunk, FaithfulnessChecker, FaithfulnessReport
from src.generation.llm_client import get_llm
from src.retrieval.policy_engine import RetrievalResult


@dataclass(frozen=True)
class GenerationResult:
    """Final answer plus the context it was grounded in and its faithfulness report."""

    query: str
    answer: str
    context: list[ContextChunk]
    faithfulness: FaithfulnessReport | None


def _build_context(
    result: RetrievalResult, config: GenerationConfig = DEFAULT_GENERATION_CONFIG
) -> list[ContextChunk]:
    """Turn the policy engine's ranked survivors into LLM context.

    `result.candidates` is already ranked by adjusted (policy-weighted)
    score and capped at the retrieval layer's own `top_k` -- this further
    caps to `generation_max_context_chunks` (independently sweepable) and
    truncates each chunk's text so one long chunk can't crowd out the rest
    of the prompt.
    """
    return [
        ContextChunk(
            chunk_id=gated.candidate.chunk_id,
            text=gated.candidate.text[: config.generation_max_chars_per_chunk],
        )
        for gated in result.candidates[: config.generation_max_context_chunks]
    ]


def _build_messages(
    query: str, context: list[ContextChunk], config: GenerationConfig
) -> list[SystemMessage | HumanMessage]:
    context_block = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in context)
    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, citing the supporting chunk id(s) "
        "in square brackets after each claim."
    )
    return [
        SystemMessage(content=config.generation_system_prompt),
        HumanMessage(content=user_content),
    ]


_faithfulness_checker: FaithfulnessChecker | None = None


def _default_faithfulness_checker(config: FaithfulnessConfig) -> FaithfulnessChecker:
    global _faithfulness_checker
    if _faithfulness_checker is None:
        _faithfulness_checker = FaithfulnessChecker(config)
    return _faithfulness_checker


def generate(
    query: str,
    result: RetrievalResult,
    *,
    config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
    faithfulness_config: FaithfulnessConfig = DEFAULT_FAITHFULNESS_CONFIG,
    llm: BaseChatModel | None = None,
    faithfulness_checker: FaithfulnessChecker | None = None,
    run_faithfulness_check: bool = True,
) -> GenerationResult:
    """Generate an answer to `query` from `result`'s gated, ranked context.

    Short-circuits without an LLM call (and without a faithfulness check --
    there's nothing to check) when the policy engine left no survivors, since
    a context-free call would only invite the model to answer from its own
    unverified knowledge.
    """
    context = _build_context(result, config)
    if not context:
        return GenerationResult(
            query=query, answer=config.generation_no_context_answer, context=[], faithfulness=None
        )

    llm = llm or get_llm(temperature=config.generation_temperature)
    messages = _build_messages(query, context, config)
    response = llm.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)

    faithfulness_report = None
    if run_faithfulness_check:
        checker = faithfulness_checker or _default_faithfulness_checker(faithfulness_config)
        faithfulness_report = checker.check(answer, context)

    return GenerationResult(
        query=query, answer=answer, context=context, faithfulness=faithfulness_report
    )


class GenerationState(TypedDict):
    """LangGraph state for the retrieve -> generate graph.

    `query`/`result` deliberately match `src.retrieval.policy_engine.
    RetrievalState`'s fields exactly, so `retrieve_node`'s partial state
    update composes directly into this larger schema (see `src/api/main.py`).
    """

    query: str
    result: RetrievalResult | None
    generation: GenerationResult | None


def generate_node(state: GenerationState) -> dict:
    """Thin LangGraph adapter around `generate`.

    Usable as `graph.add_node("generate", generate_node)`; expects
    `state["result"]` to already be populated by a prior `retrieve` node.
    """
    result = state.get("result")
    if result is None:
        raise ValueError(
            "generate_node requires state['result'] to be populated by retrieve_node first"
        )
    return {"generation": generate(state["query"], result)}
