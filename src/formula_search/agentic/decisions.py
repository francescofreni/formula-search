"""Bounded response recovery; only validated decisions advance a session."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .fireworks import IncompleteResponse

if TYPE_CHECKING:
    from .common import SearchSession

EventSink = Callable[[dict[str, Any]], None]


@dataclass
class DecisionStats:
    model_calls: int = 0
    decision_retries: int = 0
    automatic_steps: int = 0


def _request(
    session: SearchSession,
    complete: Callable[[str], str],
    prompt: str,
    attempt: int,
    emit: EventSink,
) -> tuple[str | None, str | None]:
    """Catch response errors only, never errors applying a valid proof."""
    session.decision_stats.model_calls += 1
    emit(
        {
            "event": "request",
            "attempt": attempt,
            "prompt": prompt,
            "prompt_chars": len(prompt),
        }
    )
    try:
        response = complete(prompt)
    except IncompleteResponse as exc:
        if exc.finish_reason != "length":
            raise
        emit(
            {
                "event": "incomplete_response",
                "attempt": attempt,
                "error": str(exc),
                "finish_reason": exc.finish_reason,
            }
        )
        return None, (
            str(exc) + " "
            "The previous attempt exhausted its response budget without a "
            "complete decision. Make a best-effort selection from the offered "
            "IDs, keep deliberation brief, and return the selection JSON now."
        )
    emit({"event": "response", "attempt": attempt, "response": response})
    try:
        session.validate(response)
    except ValueError as exc:
        return response, str(exc)
    return response, None


def advance(
    session: SearchSession,
    complete: Callable[[str], str],
    *,
    max_decision_retries: int = 2,
    on_event: EventSink | None = None,
) -> dict[str, Any]:
    """Apply one layer/move, with at most two corrective retries by default.

    Retries keep the exact candidates, model settings, and original prompt
    context. Only the latest feedback is appended. Forced selections skip
    the model but still count as one search step. No fallback ranks or fills
    candidates. HTTP/network retries remain the provider client's concern.
    """
    if type(max_decision_retries) is not int or max_decision_retries < 0:
        raise ValueError("max_decision_retries must be an integer >= 0")
    if session.exhausted:
        raise ValueError("Search is exhausted")

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    forced = session.forced_response()
    if forced is not None:
        session.validate(forced)
        emit(
            {
                "event": "automatic",
                "response": forced,
                "reason": "All offered choices must be retained",
            }
        )
        entry = session.step(forced)
        session.decision_stats.automatic_steps += 1
        return entry
    original = json.loads(session.prompt())
    prompt = json.dumps(original, sort_keys=True)
    for attempt in range(1, max_decision_retries + 2):
        response, error = _request(session, complete, prompt, attempt, emit)
        if error is None:
            assert response is not None
            return session.step(response)
        emit({"event": "invalid_decision", "attempt": attempt, "error": error})
        if attempt > max_decision_retries:
            raise ValueError(
                f"Decision failed after {attempt} attempts: {error}"
            )
        session.decision_stats.decision_retries += 1
        emit(
            {
                "event": "decision_retry",
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "error": error,
            }
        )
        prompt = json.dumps(
            {
                **original,
                "correction": {
                    "error": error,
                    "previous_response": response,
                    "instruction": (
                        "The search state has not changed. Return only a valid "
                        "selection using the offered option IDs. Do not repeat "
                        "IDs. Meet the required selection count exactly."
                    ),
                },
            },
            sort_keys=True,
        )
    raise AssertionError("Unreachable decision loop exit")
