"""Phase 8 planner: decomposes a question into sub-queries for the parallel
retriever-agent fan-out in `src/agentic/graph.py`.

Goes through `get_llm()` (never a provider SDK directly, per CLAUDE.md's hard
rule) -- unlike `src/detection/llm_judge.py`'s OpenAI-native
`response_format={"type": "json_object"}` structured output, which is only
available there because that module calls the OpenAI SDK directly as a
documented standalone comparison baseline. `get_llm()` returns a
provider-agnostic LangChain chat model with no equivalent guarantee, so this
module prompts for strict-JSON-only output and parses defensively.
"""

from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agentic.config import DEFAULT_CONFIG, AgenticConfig
from src.generation.llm_client import get_llm

PLANNER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a query-planning agent for a retrieval system. Decide whether "
    "the user's question is a single atomic factual lookup or genuinely has "
    "multiple distinct parts needing separate lookups (e.g. comparing two "
    "entities, or requiring two independent facts). If it is atomic, return "
    "exactly one sub-query equal to the original question. If it has "
    "multiple parts, decompose it into 2-{max_subqueries} independent, "
    "self-contained sub-queries. Respond with strict JSON only, no "
    'markdown, no commentary: {{"sub_queries": ["...", ...]}}'
)


def plan_subqueries(
    query: str,
    *,
    config: AgenticConfig = DEFAULT_CONFIG,
    llm: BaseChatModel | None = None,
) -> list[str]:
    """Decompose `query` into up to `config.agentic_max_subqueries` sub-queries.

    Never raises: any LLM-call failure, unparseable/malformed response, or a
    resulting deduplicated sub-query count below 2 falls back to `[query]`
    verbatim (the literal original text, not a single LLM-rephrased variant)
    -- an atomic/single-hop question's agentic behavior then stays exactly
    comparable to the un-decomposed Phase 4/5 baseline. This is the expected,
    common outcome on Phase 1's single-hop NQ corpus, not a degraded case.
    """
    try:
        active_llm = llm or get_llm(temperature=config.agentic_planner_temperature)
        messages = [
            SystemMessage(
                content=PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
                    max_subqueries=config.agentic_max_subqueries
                )
            ),
            HumanMessage(content=query),
        ]
        response = active_llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = json.loads(content)

        seen: set[str] = set()
        sub_queries: list[str] = []
        for raw in parsed.get("sub_queries", []):
            candidate = str(raw).strip()
            if len(candidate) >= config.agentic_min_subquery_chars and (
                candidate.lower() not in seen
            ):
                seen.add(candidate.lower())
                sub_queries.append(candidate)
        sub_queries = sub_queries[: config.agentic_max_subqueries]
    except Exception:  # noqa: BLE001 -- planner failure must never abort the pipeline
        sub_queries = []

    return sub_queries if len(sub_queries) >= 2 else [query]
