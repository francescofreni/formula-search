"""Fireworks callback using its Chat Completions HTTP API (no extra SDK)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


class Fireworks:
    """Choose an offered move with schema-constrained JSON output.

    Model: explicit argument or FIREWORKS_MODEL. Key: explicit argument,
    FIREWORKS_API_KEY, then ~/.config/fireworksai/api_key. No API call occurs
    until __call__. Credentials are never put in prompts or search history.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 120,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("FIREWORKS_MODEL", "")
        if not self.model:
            raise ValueError("Supply a model or set FIREWORKS_MODEL")
        key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        if not key:
            path = Path.home() / ".config/fireworksai/api_key"
            key = path.read_text().strip() if path.is_file() else ""
        if not key.strip():
            raise ValueError("Set FIREWORKS_API_KEY or supply an api_key")
        self._key = key.strip()
        self.timeout = timeout
        if max_tokens is not None and (
            type(max_tokens) is not int or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def complete(self, prompt: str) -> str:
        """Send a plain prompt and return text, without a move schema."""
        return self._request(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
        )

    def __call__(self, prompt: str) -> str:
        state = json.loads(prompt)
        options = state["options"]
        if not options:
            raise ValueError("No legal moves to choose from")
        ids = [option["id"] for option in options]
        field = "keep" if state.get("strategy") == "beam" else "move"
        choice: dict[str, Any] = {"type": "integer", "enum": ids}
        if field == "keep":
            count = state["selection_count"]
            choice = {
                "type": "array",
                "items": choice,
                "minItems": count,
                "maxItems": count,
                "uniqueItems": True,
            }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "BeamSelection" if field == "keep" else "Move",
                    "schema": {
                        "type": "object",
                        "properties": {field: choice},
                        "required": [field],
                        "additionalProperties": False,
                    },
                },
            },
        }
        return self._request(payload)

    def _request(self, payload: dict[str, Any]) -> str:
        if payload.get("max_tokens") is None:
            payload.pop("max_tokens", None)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        request = Request(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        choice = result["choices"][0]
        text = choice["message"].get("content")
        reason = choice.get("finish_reason")
        if reason != "stop" or not isinstance(text, str) or not text.strip():
            usage = result.get("usage", {})
            detail = (
                f"finish_reason={reason!r}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"max_tokens={payload.get('max_tokens', 'provider default')}, "
                f"content_chars={len(text) if isinstance(text, str) else 0}"
            )
            hint = (
                " Reasoning can consume the budget before a final answer. "
                "For models supporting it, try --reasoning-effort none. "
                "Increase --max-tokens only if more reasoning is intended."
                if reason == "length"
                else ""
            )
            raise ValueError(
                "Fireworks did not return a complete decision or text "
                f"response ({detail}).{hint}"
            )
        return text
