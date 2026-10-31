"""Readable work budgets, independent of search dependencies."""

from typing import Any


def budget(case: dict[str, Any]) -> str:
    s = case["strategy"]
    if s["method"] == "derivation":
        return f"cost bound {s['bound']}"
    if s["method"] == "astar":
        return f"{s['max_pops']} queue pops"
    if s["method"] == "beam":
        return f"{s['max_layers']} layers; width {s['width']}"
    return f"{s['max_steps']} moves"
