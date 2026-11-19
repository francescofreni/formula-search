"""Adapt any model to deterministic depth-first/beam decision schemas."""

import json
from typing import Any

from .model import Model


class DecisionModel:
    """Wrap prompt completion with the schema derived from offered options."""

    def __init__(self, model: Model) -> None:
        self.model = model

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
        return self.model.complete(
            prompt,
            response_format={
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
        )
