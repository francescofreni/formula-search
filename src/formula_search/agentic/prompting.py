"""Prompt presentation without changing candidates, order, or instructions."""

import json
from typing import Any

BEAM_LAYOUT = "beam-parents-v1"
DEPTH_FIRST_LAYOUT = "depth-first-v1"


def beam_context(options: list[dict[str, Any]]) -> dict[str, Any]:
    """Store each parent equation/path once; retain every candidate field.

    Replacing parent_id with its parent's current_equation and branch_path
    reconstructs the old option exactly. No history, candidate, or variable
    scope is truncated or summarized by a model.
    """
    parents: list[dict[str, Any]] = []
    indices: dict[str, int] = {}
    compact = []
    for option in options:
        parent = {
            "current_equation": option["from_equation"],
            "branch_path": option["branch_path"],
        }
        key = json.dumps(parent, sort_keys=True)
        if key not in indices:
            indices[key] = len(parents)
            parents.append({"id": indices[key], **parent})
        compact.append(
            {
                "parent_id": indices[key],
                **{
                    k: v
                    for k, v in option.items()
                    if k not in ("from_equation", "branch_path")
                },
            }
        )
    return {
        "prompt_layout": BEAM_LAYOUT,
        "layout_help": (
            "Each option's parent_id refers to an entry in parents, where "
            "its starting equation and full branch_path are shown once. "
            "The option shows the move and resulting equation. Select "
            "option IDs, never parent IDs, across all parents together."
        ),
        "parents": parents,
        "options": compact,
    }
