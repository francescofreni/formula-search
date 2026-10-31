"""Human-readable tables and machine-readable rows, including incomplete runs."""

from __future__ import annotations

import csv
import fcntl
import io
from collections import Counter
from pathlib import Path
from typing import Any

from .budgets import budget
from .storage import atomic_text, now, read_json


def rows(out: Path) -> list[dict[str, Any]]:
    result = []
    for case in read_json(out / "cases.json"):
        data = read_json(out / case["path"] / "result.json")
        strategy = case["strategy"]
        result.append(
            {
                "task": case["task"]["name"],
                "strategy": strategy["name"],
                "method": strategy["method"],
                "repeat": case["repeat"],
                "budget": budget(case),
                "status": data["status"],
                "classes": (
                    len(data["classes"]) if data["counts_available"] else ""
                ),
                "completed_actions": data["completed_actions"],
                **{
                    k: data.get(k, "")
                    for k in (
                        "model_calls",
                        "decision_retries",
                        "automatic_steps",
                        "api_attempt_retries",
                    )
                },
                "prompt_layout": strategy.get("prompt_layout", ""),
                "elapsed_seconds": round(data["elapsed_seconds"], 3),
                "prompt_version": strategy.get("prompt_version", ""),
                "prompt_sha256": strategy.get("prompt_sha256", ""),
                "model": strategy.get("model", ""),
                "profile": strategy.get("profile", {}).get("name", ""),
                "source_sha256": read_json(out / "manifest.json")["source"][
                    "sha256"
                ],
                "error": data["error"] or "",
                "result": f"{case['path']}/result.json",
            }
        )
    return result


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(out: Path) -> str:
    # Only the aggregate report is shared; each worker owns its own case files.
    with (out / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = rows(out)
        spec = read_json(out / "experiment.json")
        states = Counter(row["status"] for row in data)
        text = f"# {spec['name']}\n\n{spec['description']}\n\nUpdated: {now()}\n\n"
        text += (
            ", ".join(
                f"{count} {status}" for status, count in sorted(states.items())
            )
            + ".\n\n"
        )
        text += (
            "Counts are distinct verified formula classes within each run, not novel "
            "relative to other strategies. Budgets measure different kinds of work; "
            "this is not an equal-compute comparison. Errors retain partial counts. "
            "`running` is a last saved state, not proof that a process is still alive; "
            "check Slurm for hard kills. Model calls include corrective attempts; "
            "automatic steps advance the search without calling a model. "
            "Transport retries are counted separately in the CSV.\n\n"
            "| Task | Strategy | Repeat | Budget | Classes | Actions | Model calls | Auto steps | Response retries | Seconds | Status | Details |\n"
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        )
        for row in data:
            folder = str(Path(row["result"]).parent)
            text += (
                "| "
                + " | ".join(
                    cell(row[k])
                    for k in (
                        "task",
                        "strategy",
                        "repeat",
                        "budget",
                        "classes",
                        "completed_actions",
                        "model_calls",
                        "automatic_steps",
                        "decision_retries",
                        "elapsed_seconds",
                        "status",
                    )
                )
                + f" | [details]({folder}/README.md) |\n"
            )
        text += (
            "\n## Saved configuration\n\n"
            "[Experiment](experiment.json) · [Exact cases](cases.json) · "
            "[CSV](summary.csv) · [Source/dependencies](manifest.json) · "
            "[Replay instructions](REPLAY.md) · [Slurm logs](slurm/)\n\n"
            "Refresh this table with `python experiments/run.py report RUN_FOLDER`.\n"
        )
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
        atomic_text(out / "summary.csv", buffer.getvalue())
        atomic_text(out / "README.md", text)
        return text


def history(experiments: Path) -> str:
    text = "# Experiment history\n\n"
    for p in sorted(
        (experiments / "results").glob("*/manifest.json"), reverse=True
    ):
        manifest = read_json(p)
        if manifest.get("schema") != "formula-experiment-v1":
            continue
        out = p.parent
        states = Counter(r["status"] for r in rows(out))
        text += f"- [{out.name}]({out.relative_to(experiments)}/README.md): {dict(states)}\n"
    text += "\n## Earlier experiment layouts\n\nOriginal files are retained in place.\n\n"
    for result_dir in sorted(experiments.rglob("results")):
        if (
            result_dir == experiments / "results"
            or "source" in result_dir.parts
        ):
            continue
        jsons = list(result_dir.rglob("result.json"))
        if not jsons:
            jsons = list(result_dir.rglob("*.json"))
        text += f"- [{result_dir.relative_to(experiments)}]({result_dir.relative_to(experiments)}/): {len(jsons)} result/JSON files\n"
    atomic_text(experiments / "HISTORY.md", text)
    return text
