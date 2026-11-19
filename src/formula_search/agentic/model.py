"""Provider-independent prompt-to-text contract and response errors."""

from typing import Any, Protocol


class Model(Protocol):
    """A stateless completion client; callers own conversation/search state.

    Providers must return complete, nonempty text or raise an error. A
    requested response format must be forwarded or explicitly rejected;
    silently dropping it changes the search experiment. The search still
    validates every decision locally, even with a requested JSON schema.
    """

    def complete(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Return text, optionally requesting a structured response format."""
        ...


class IncompleteResponse(ValueError):
    """Provider returned no complete answer; distinguish response truncation."""

    def __init__(self, message: str, finish_reason: str | None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
