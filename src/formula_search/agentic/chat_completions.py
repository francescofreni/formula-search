"""Shared Chat Completions transport, credentials, retries and validation."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model import IncompleteResponse
from .selection import DecisionModel


class ChatCompletionsModel:
    """Base for providers with a Chat Completions HTTP API.

    Subclasses define provider metadata and may adapt optional payload
    fields. Construction resolves credentials but makes no network call.
    """

    provider: str
    endpoint: str
    model_env: str
    key_env: str
    key_directory: str

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 300,
        max_retries: int = 5,
        on_retry: Callable[[dict[str, Any]], None] | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model or os.environ.get(self.model_env, "")
        if not self.model:
            raise ValueError(f"Supply a model or set {self.model_env}")
        key = api_key or os.environ.get(self.key_env, "")
        if not key:
            path = Path.home() / ".config" / self.key_directory / "api_key"
            key = path.read_text().strip() if path.is_file() else ""
        if not key.strip():
            raise ValueError(f"Set {self.key_env} or supply an api_key")
        self._key = key.strip()
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_retry = on_retry
        if max_tokens is not None and (
            type(max_tokens) is not int or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def complete(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send arbitrary text; search schemas are opt-in."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            # Provider adaptation must not mutate the caller's schema.
            payload["response_format"] = json.loads(
                json.dumps(response_format)
            )
        self._configure_payload(payload)
        return self._request(payload)

    def __call__(self, prompt: str) -> str:
        """Compatibility shorthand for selecting a search move or beam."""
        return DecisionModel(self)(prompt)

    def _configure_payload(self, payload: dict[str, Any]) -> None:
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

    def _request(self, payload: dict[str, Any]) -> str:
        if payload.get("max_tokens") is None:
            payload.pop("max_tokens", None)
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        result = self._send(request)
        try:
            choice = result["choices"][0]
            text = choice["message"].get("content")
            reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise IncompleteResponse(
                f"{self.provider} returned a malformed completion response",
                None,
            ) from exc
        if reason != "stop" or not isinstance(text, str) or not text.strip():
            usage = result.get("usage")
            if not isinstance(usage, dict):
                usage = {}
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
            raise IncompleteResponse(
                f"{self.provider} did not return a complete decision or text "
                f"response ({detail}).{hint}",
                reason,
            )
        return text

    def _retry_event(self, **event: Any) -> None:
        event["timestamp"] = time.time()
        logging.getLogger(__name__).warning(
            "%s request: %s", self.provider, event
        )
        if self.on_retry is not None:
            self.on_retry(event)

    def _send(self, request: Request) -> Any:
        """Retry transport failures without changing the search state.

        Five retries mean at most six attempts. Backoff starts at two seconds,
        doubles up to 60 seconds, and adds independent jitter. Retry-After is
        a minimum delay. Invalid responses and permanent HTTP errors propagate.
        A timed-out request may still have been processed by the provider.
        """
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
            except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
                if isinstance(exc, HTTPError):
                    if (
                        exc.code not in (408, 429)
                        and not 500 <= exc.code < 600
                    ):
                        raise
                    reason = f"HTTP {exc.code}"
                    retry_after = exc.headers.get("Retry-After")
                    exc.close()
                else:
                    reason = type(exc).__name__
                if attempt == self.max_retries:
                    self._retry_event(
                        event="exhausted", attempt=attempt + 1, reason=reason
                    )
                    raise
                delay = _retry_delay(attempt, retry_after)
                self._retry_event(
                    event="retry",
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    reason=reason,
                    delay_seconds=delay,
                )
                time.sleep(delay)
            else:
                if attempt:
                    self._retry_event(event="recovered", attempt=attempt + 1)
                return result
        raise AssertionError("Unreachable retry loop exit")


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    delay = min(60.0, 2.0 ** min(attempt + 1, 6))
    delay += random.SystemRandom().uniform(0, 1)
    if not retry_after:
        return delay
    try:
        seconds = float(retry_after)
    except ValueError:
        try:
            seconds = (
                parsedate_to_datetime(retry_after).timestamp() - time.time()
            )
        except (ValueError, TypeError, OverflowError):
            seconds = 0.0
    return max(delay, seconds) if math.isfinite(seconds) else delay
