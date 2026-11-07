"""OpenAI-compatible client construction (OpenRouter by default)."""

from __future__ import annotations

import os

from openai import AsyncOpenAI, NotFoundError

from .resilience import OPENROUTER_BASE_URL, is_openrouter_base_url

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 400
LLM_TOP_P = 1.0


def raise_model_unavailable(model: str, err: NotFoundError) -> None:
    raise RuntimeError(
        "Configured endpoint could not serve model "
        f"'{model}' (404/not found). "
        "Use a valid model ID for your selected backend and retry."
    ) from err


def create_client() -> AsyncOpenAI:
    """Build the async client from the environment (LLM_BASE_URL, OPENROUTER_API_KEY)."""
    base_url = os.getenv("LLM_BASE_URL", OPENROUTER_BASE_URL).strip()
    use_openrouter = is_openrouter_base_url(base_url)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key and not use_openrouter:
        # Ollama and other local OpenAI-compatible servers often ignore the key,
        # but the client still expects a non-empty value.
        api_key = "local"
    if not api_key:
        raise EnvironmentError(
            "Set OPENROUTER_API_KEY (or OPENAI_API_KEY) before running the planner."
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def uses_openrouter() -> bool:
    return is_openrouter_base_url(os.getenv("LLM_BASE_URL", OPENROUTER_BASE_URL))
