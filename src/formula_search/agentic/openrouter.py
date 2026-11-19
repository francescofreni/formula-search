"""OpenRouter Chat Completions adapter (no additional SDK)."""

from typing import Any

from .chat_completions import ChatCompletionsModel


class OpenRouter(ChatCompletionsModel):
    """Key: explicit argument, OPENROUTER_API_KEY, or its local key file."""

    provider = "OpenRouter"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    model_env = "OPENROUTER_MODEL"
    key_env = "OPENROUTER_API_KEY"
    key_directory = "openrouter"

    def _configure_payload(self, payload: dict[str, Any]) -> None:
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if "response_format" in payload:
            if payload["response_format"].get("type") == "json_schema":
                payload["response_format"]["json_schema"]["strict"] = True
            # Fail if no endpoint supports the requested schema; do not
            # silently route a search decision to an unconstrained model.
            payload["provider"] = {"require_parameters": True}
