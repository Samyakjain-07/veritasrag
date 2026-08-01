"""Provider-agnostic LLM client for generation.

Reads `LLM_PROVIDER` ("openai" or "anthropic", default "openai") and
returns a LangChain chat model behind one shared interface
(`langchain_core.language_models.chat_models.BaseChatModel`), so the rest
of the codebase (`src/generation/generate.py`, faithfulness checks, etc.)
never imports `langchain_openai`/`langchain_anthropic` -- or a provider SDK
directly -- itself. Swap providers via the `LLM_PROVIDER` env var without
touching any calling code.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

SUPPORTED_PROVIDERS = ("openai", "anthropic")


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: object,
) -> BaseChatModel:
    """Return a LangChain chat model for the configured (or given) provider.

    `provider` defaults to the `LLM_PROVIDER` env var ("openai" if unset),
    re-read on every call rather than cached at import time so a changed
    env var takes effect without a process restart. `model` defaults to
    that provider's configured default model. Extra `kwargs` (e.g.
    `temperature`, `api_key`) are forwarded to the underlying LangChain
    chat model constructor.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        default_model = os.getenv("OPENAI_GENERATION_MODEL", "gpt-5.4")
        return ChatOpenAI(model=model or default_model, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        default_model = os.getenv("ANTHROPIC_GENERATION_MODEL", "claude-sonnet-5")
        return ChatAnthropic(model=model or default_model, **kwargs)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r} (expected one of {SUPPORTED_PROVIDERS})"
    )
