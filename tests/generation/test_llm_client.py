import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from src.generation.llm_client import get_llm


def test_get_llm_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    llm = get_llm(api_key="sk-fake-test-key")

    assert isinstance(llm, ChatOpenAI)


def test_get_llm_openai_provider_explicit():
    llm = get_llm(provider="openai", api_key="sk-fake-test-key")

    assert isinstance(llm, ChatOpenAI)


def test_get_llm_anthropic_provider_explicit():
    llm = get_llm(provider="anthropic", api_key="sk-ant-fake-test-key")

    assert isinstance(llm, ChatAnthropic)


def test_get_llm_reads_llm_provider_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    llm = get_llm(api_key="sk-ant-fake-test-key")

    assert isinstance(llm, ChatAnthropic)


def test_get_llm_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm(provider="not-a-real-provider")


def test_get_llm_model_override_is_forwarded():
    llm = get_llm(provider="openai", model="gpt-5.4-mini", api_key="sk-fake-test-key")

    assert llm.model_name == "gpt-5.4-mini"
