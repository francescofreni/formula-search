"""Named provider construction for command-line and experiment configs."""

from collections.abc import Callable
from typing import Any

from .chat_completions import ChatCompletionsModel
from .fireworks import Fireworks
from .openrouter import OpenRouter

PROVIDERS: dict[str, type[ChatCompletionsModel]] = {
    "fireworks": Fireworks,
    "openrouter": OpenRouter,
}


def create_model(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    timeout: float = 300,
    max_retries: int = 5,
    on_retry: Callable[[dict[str, Any]], None] | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> ChatCompletionsModel:
    """Construct a provider explicitly; never infer it from a model ID."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown model provider: {provider!r}")
    return PROVIDERS[provider](
        model,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        on_retry=on_retry,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
