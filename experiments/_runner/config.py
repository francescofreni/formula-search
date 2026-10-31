"""Resolve human-written experiment definitions into explicit independent cases."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from formula_search.agentic.prompting import BEAM_LAYOUT, DEPTH_FIRST_LAYOUT
from formula_search.agentic.prompts import PROMPT_VERSION, get_prompt
from formula_search.baselines.id import parse_query
from formula_search.graph import ADMG

from .profiles import COSTS, PROFILES, TASKS

PROFILES_BY_NAME = {p.name: p for p in PROFILES}
METHODS = ("derivation", "astar", "depth-first", "beam")


def slug(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]*", value
    ):
        raise ValueError(
            f"Use lowercase letters, digits, '-' or '_' in names: {value!r}"
        )
    return value


def keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a JSON object")
    extra = data.keys() - allowed
    if extra:
        raise ValueError(f"Unknown {context} settings: {sorted(extra)}")


def integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def task_spec(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        if value not in TASKS:
            raise ValueError(f"Unknown task {value}; choose {list(TASKS)}")
        task = TASKS[value]
        return {
            "name": value,
            "graph": task.graph,
            "treatment": task.treatment,
            "outcome": task.outcome,
        }
    keys(value, {"name", "graph", "treatment", "outcome"}, "task")
    slug(value["name"])
    graph = ADMG.parse(value["graph"])
    for field in ("treatment", "outcome"):
        if (
            not isinstance(value[field], str)
            or value[field] not in graph.nodes
        ):
            raise ValueError(f"{field} must name a single graph variable")
    parse_query(graph, value["treatment"], value["outcome"])
    return dict(value)


def strategy_spec(raw: dict[str, Any]) -> dict[str, Any]:
    method = raw["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose {METHODS}")
    allowed = {
        "derivation": {"bound"},
        "astar": {"profile", "max_pops"},
        "depth-first": {"max_steps"},
        "beam": {"width", "max_layers"},
    }[method]
    agentic = method in ("depth-first", "beam")
    if agentic:
        allowed |= {
            "model",
            "prompt_version",
            "reasoning_effort",
            "api_timeout",
            "max_retries",
            "decision_retries",
        }
    keys(raw, allowed | {"name", "method", "repeats"}, "strategy")
    spec = {
        "name": slug(raw.get("name", method)),
        "method": method,
        "repeats": integer(raw.get("repeats", 3 if agentic else 1), "repeats"),
    }
    if method == "derivation":
        spec["bound"] = integer(raw.get("bound", 4), "bound", 0)
    elif method == "astar":
        profile = raw.get("profile", "lookahead")
        if profile not in PROFILES_BY_NAME:
            raise ValueError(f"Unknown A* profile {profile}")
        spec.update(
            profile=asdict(PROFILES_BY_NAME[profile]),
            max_pops=integer(raw.get("max_pops", 1000), "max_pops"),
        )
    if agentic:
        spec.update(agent_settings(raw, method))
    return spec


def agent_settings(raw: dict[str, Any], method: str) -> dict[str, Any]:
    template = get_prompt(method, raw.get("prompt_version", PROMPT_VERSION))
    spec = {
        **template.metadata,
        "prompt_layout": (
            BEAM_LAYOUT if method == "beam" else DEPTH_FIRST_LAYOUT
        ),
        "decision_retries": integer(
            raw.get("decision_retries", 2), "decision_retries", 0
        ),
        "model": raw.get("model", "accounts/fireworks/models/glm-5p3-flash"),
        "reasoning_effort": raw.get("reasoning_effort", "low"),
        "api_timeout": integer(raw.get("api_timeout", 300), "api_timeout"),
        "max_retries": integer(raw.get("max_retries", 5), "max_retries", 0),
        "max_tokens": None,
        "temperature": 0,
        "api_endpoint": "https://api.fireworks.ai/inference/v1/chat/completions",
    }
    if not isinstance(spec["model"], str) or not spec["model"].strip():
        raise ValueError("model must be a nonempty string")
    if method == "beam":
        spec.update(
            width=integer(raw.get("width", 10), "width"),
            max_layers=integer(raw.get("max_layers", 50), "max_layers"),
        )
    else:
        spec["max_steps"] = integer(raw.get("max_steps", 200), "max_steps")
    return spec


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(path.read_text())
    keys(
        raw,
        {
            "name",
            "description",
            "tasks",
            "strategies",
            "verifier_seed",
            "slurm",
        },
        "experiment",
    )
    tasks = [task_spec(t) for t in raw["tasks"]]
    strategies = [strategy_spec(s) for s in raw["strategies"]]
    for kind, items in (("task", tasks), ("strategy", strategies)):
        if not items or len({x["name"] for x in items}) != len(items):
            raise ValueError(f"Supply nonempty, uniquely named {kind}s")
    scheduler = {
        "concurrency": 2,
        "time": "7-00:00:00",
        "memory": "8G",
        **raw.get("slurm", {}),
    }
    keys(scheduler, {"concurrency", "time", "memory"}, "Slurm")
    integer(scheduler["concurrency"], "concurrency")
    if not re.fullmatch(r"(?:\d+-)?\d+:\d{2}:\d{2}", scheduler["time"]):
        raise ValueError("Slurm time must be [days-]hours:minutes:seconds")
    if not re.fullmatch(r"[1-9]\d*[MG]", scheduler["memory"]):
        raise ValueError("Slurm memory must be e.g. 8G or 512M")
    spec = {
        "name": slug(raw["name"]),
        "description": raw.get("description", ""),
        "tasks": tasks,
        "strategies": strategies,
        "verifier_seed": integer(
            raw.get("verifier_seed", 0), "verifier_seed", 0
        ),
        "slurm": scheduler,
    }
    cases = []
    for task in tasks:
        for strategy in strategies:
            for repeat in range(1, strategy["repeats"] + 1):
                cases.append(
                    {
                        "id": len(cases),
                        "task": task,
                        "strategy": strategy,
                        "repeat": repeat,
                        "verifier_seed": spec["verifier_seed"],
                        "costs": COSTS,
                        "prune_duplicates": True,
                        "path": f"{task['name']}/{strategy['name']}/repeat-{repeat:02d}",
                    }
                )
    return spec, cases
