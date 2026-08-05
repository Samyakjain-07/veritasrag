from src.agentic.config import AgenticConfig
from src.agentic.planner import plan_subqueries


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
        raise RuntimeError("simulated API failure (e.g. rate limit)")


def test_plan_subqueries_decomposes_multi_part_question():
    llm = _FakeLLM('{"sub_queries": ["Who directed Inception?", "Who directed Interstellar?"]}')

    sub_queries = plan_subqueries("Who directed Inception and who directed Interstellar?", llm=llm)

    assert sub_queries == ["Who directed Inception?", "Who directed Interstellar?"]


def test_plan_subqueries_falls_back_to_original_on_atomic_question():
    llm = _FakeLLM('{"sub_queries": ["What is the capital of France?"]}')

    sub_queries = plan_subqueries("What is the capital of France?", llm=llm)

    assert sub_queries == ["What is the capital of France?"]


def test_plan_subqueries_falls_back_to_original_on_malformed_json():
    llm = _FakeLLM("not valid json at all")

    sub_queries = plan_subqueries("some question", llm=llm)

    assert sub_queries == ["some question"]


def test_plan_subqueries_falls_back_to_original_on_llm_exception():
    sub_queries = plan_subqueries("some question", llm=_ExplodingLLM())

    assert sub_queries == ["some question"]


def test_plan_subqueries_dedupes_case_insensitively():
    llm = _FakeLLM(
        '{"sub_queries": ["Who wrote Hamlet?", "who wrote hamlet?", "When was Hamlet written?"]}'
    )

    sub_queries = plan_subqueries("question", llm=llm)

    assert sub_queries == ["Who wrote Hamlet?", "When was Hamlet written?"]


def test_plan_subqueries_caps_at_configured_max():
    llm = _FakeLLM('{"sub_queries": ["a query", "b query", "c query", "d query"]}')
    config = AgenticConfig(agentic_max_subqueries=2)

    sub_queries = plan_subqueries("question", config=config, llm=llm)

    assert sub_queries == ["a query", "b query"]


def test_plan_subqueries_drops_too_short_entries():
    llm = _FakeLLM('{"sub_queries": ["ok", "a valid sub-query", "another valid sub-query"]}')
    config = AgenticConfig(agentic_min_subquery_chars=5)

    sub_queries = plan_subqueries("question", config=config, llm=llm)

    assert sub_queries == ["a valid sub-query", "another valid sub-query"]
