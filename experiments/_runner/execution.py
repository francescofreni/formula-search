"""One case, shared verification and artifacts, strategy-specific work budgets."""

from __future__ import annotations

import json
import os
import random
import signal
import socket
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from formula_search.agentic import BeamSession, DepthFirstSession, Fireworks
from formula_search.agentic.common import SearchSession
from formula_search.agentic.decisions import advance
from formula_search.astar import SearchStats, _Search
from formula_search.derivation import Derivation, derive
from formula_search.evaluator import Archive, Task
from formula_search.graph import ADMG

from .budgets import budget
from .config import PROFILES_BY_NAME
from .profiles import COSTS, configured
from .report import render
from .storage import atomic_text, encode, now, read_json, write_json


class DuplicateRun(FileExistsError):
    """This attempt did not acquire the case and must not change its results."""


class Interrupted(Exception):
    pass


def stop(signum: int, frame: Any) -> None:
    raise Interrupted(f"Received signal {signum}")


class Recorder:
    def __init__(self, out: Path, case: dict[str, Any]) -> None:
        self.out = out
        self.case = case
        self.task = Task(
            case["task"]["graph"],
            case["task"]["treatment"],
            case["task"]["outcome"],
        )
        self.archive = Archive(self.task)
        self.start = time.perf_counter()
        self.stats = SearchStats()
        self.processed_verdicts = 0
        self.data: dict[str, Any] = {
            "schema": "formula-case-v1",
            "status": "running",
            "error": None,
            "started_utc": now(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "elapsed_seconds": 0,
            "classes": [],
            "discoveries": [],
            "completed_actions": 0,
            "counts_available": case["strategy"]["method"] != "derivation",
            "budget": budget(case),
            "api_attempt_retries": 0,
        }

    def log(self, message: str) -> None:
        print(message, flush=True)
        with (self.out / "run.log").open("a") as stream:
            stream.write(message + "\n")

    def event(self, filename: str, data: Any) -> None:
        with (self.out / filename).open("a") as stream:
            stream.write(json.dumps(data, default=encode) + "\n")

    def collect(self) -> None:
        for verdict in self.archive.log[self.processed_verdicts :]:
            if not verdict.certified or verdict.outcome not in (
                "new",
                "duplicate",
            ):
                raise RuntimeError(
                    f"Invalid proof submission: {verdict.outcome}"
                )
            if verdict.outcome == "new":
                assert isinstance(verdict.source, Derivation)
                self.data["discoveries"].append(
                    {
                        "class_index": verdict.class_index,
                        "formula": verdict.formula,
                        "action": self.data["completed_actions"],
                        "elapsed_seconds": time.perf_counter() - self.start,
                        "certified": True,
                        "derivation": asdict(verdict.source),
                    }
                )
                self.log(
                    f"New class {len(self.data['discoveries'])}: {verdict.formula}"
                )
        self.processed_verdicts = len(self.archive.log)
        self.data["classes"] = [c.formula for c in self.archive.classes]

    def save(self) -> None:
        self.data["elapsed_seconds"] = time.perf_counter() - self.start
        write_json(self.out / "result.json", self.data)
        s = self.case["strategy"]
        text = (
            f"# {self.case['task']['name']} / {s['name']} / repeat {self.case['repeat']}\n\n"
            f"Status: **{self.data['status']}** · Budget: {budget(self.case)}\n\n"
            f"Graph: `{self.task.graph}`\n\nTarget: `{self.task.target}`\n\n"
            f"Actions completed: {self.data['completed_actions']}\n\n"
            f"Elapsed: {self.data['elapsed_seconds']:.3f} seconds\n\n"
            f"Error: {self.data['error'] or 'none'}\n\n"
            "## Settings\n\n```json\n" + json.dumps(s, indent=2) + "\n```\n\n"
            "## Verified equations\n\n"
        )
        if not self.data["counts_available"]:
            text += (
                "Unavailable: bounded derivation has not returned its batch.\n"
            )
        elif not self.data["classes"]:
            text += "None found yet.\n"
        else:
            text += (
                "\n".join(
                    f"{i + 1}. `{formula}`"
                    for i, formula in enumerate(self.data["classes"])
                )
                + "\n"
            )
        text += "\n[Exact settings](config.json) · [Results and proofs](result.json) · [Progress log](run.log)\n"
        if s["method"] in ("depth-first", "beam"):
            text += (
                f"\nModel calls: {self.data.get('model_calls', 0)} · "
                f"Automatic steps: {self.data.get('automatic_steps', 0)} · "
                f"Response retries: {self.data.get('decision_retries', 0)}\n"
            )
            text += "\n[Prompt](prompt.txt) · [Decisions](decisions.jsonl) · [Retries](retries.jsonl)\n"
        atomic_text(self.out / "README.md", text)


def deterministic(record: Recorder) -> None:
    spec = record.case["strategy"]
    if spec["method"] == "derivation":
        record.log(
            "Bounded derivation returns a batch; archival progress follows search completion."
        )
        proofs = derive(*record.task, bound=spec["bound"], costs=COSTS)
        record.data["counts_available"] = True
        for proof in proofs:
            record.archive.submit(proof.formula, proof)
            record.collect()
            record.save()
        record.data["status"] = "exhausted"
        return
    profile = PROFILES_BY_NAME[spec["profile"]["name"]]
    heuristic, priority, tie = configured(record.task.graph, profile)
    search = _Search(
        ADMG.parse(record.task.graph),
        costs=COSTS,
        heuristic=heuristic,
        priority=priority,
        tiebreaker=tie,
    )
    iterator = search.run(
        record.task.query, None, max_pops=spec["max_pops"], stats=record.stats
    )
    try:
        for proof in iterator:
            record.data["completed_actions"] = record.stats.queue_pops
            record.archive.submit(proof.formula, proof)
            record.collect()
            record.save()
    finally:
        iterator.close()
        record.data["completed_actions"] = record.stats.queue_pops
        record.data["queue_stats"] = asdict(record.stats)
    record.data["status"] = record.stats.stop_reason


def decision_event(
    record: Recorder, session: SearchSession, step: int, event: dict[str, Any]
) -> None:
    record.data.update(asdict(session.decision_stats))
    record.event("decisions.jsonl", {"step": step, **event})
    if event["event"] in ("decision_retry", "incomplete_response"):
        record.event("retries.jsonl", {"step": step, **event})
        record.log(f"Decision {step}: {event['event']}: {event['error']}")
    elif event["event"] == "request":
        record.log(
            f"Decision {step}, attempt {event['attempt']}: "
            f"{event['prompt_chars']} prompt characters"
        )
    elif event["event"] == "automatic":
        record.log(f"Decision {step}: forced selection; no model request")
    record.save()


def agentic(record: Recorder) -> None:
    spec = record.case["strategy"]
    kwargs = {"costs": COSTS, "prompt_version": spec["prompt_version"]}
    session = (
        BeamSession(record.task, beam_width=spec["width"], **kwargs)
        if spec["method"] == "beam"
        else DepthFirstSession(record.task, **kwargs)
    )
    record.archive = session.archive
    if session.prompt_template.sha256 != spec["prompt_sha256"]:
        raise ValueError("Prompt differs from the planned hash")
    (record.out / "prompt.txt").write_text(
        session.prompt_template.instructions, encoding="utf-8"
    )
    for filename in ("decisions.jsonl", "retries.jsonl"):
        (record.out / filename).touch()
    step = 0

    def retry(event: dict[str, Any]) -> None:
        record.event("retries.jsonl", {"step": step, **event})
        record.log(f"API decision {step}: {event}")
        if event["event"] == "retry":
            record.data["api_attempt_retries"] += 1

    client = Fireworks(
        spec["model"],
        timeout=spec["api_timeout"],
        max_retries=spec["max_retries"],
        reasoning_effort=spec["reasoning_effort"],
        on_retry=retry,
    )
    cap = spec["max_layers"] if spec["method"] == "beam" else spec["max_steps"]
    try:
        for step in range(1, cap + 1):
            if session.exhausted:
                break
            entry = advance(
                session,
                client,
                max_decision_retries=spec["decision_retries"],
                on_event=partial(decision_event, record, session, step),
            )
            record.event("decisions.jsonl", {"step": step, "applied": entry})
            record.data["completed_actions"] = step
            record.data.update(asdict(session.decision_stats))
            record.collect()
            record.log(
                f"{'Layer' if spec['method'] == 'beam' else 'Step'} {step}: "
                f"{len(record.archive.classes)} classes"
            )
            selected = entry.get("selected", [entry])
            for move in selected:
                record.log(f"  {move['rule']}: {move['resulting_equation']}")
            record.save()
        record.data["status"] = (
            "exhausted" if session.exhausted else "step_limit"
        )
    finally:
        record.data.update(asdict(session.decision_stats))
        record.data["pruned"] = session.pruned


def execute(out: Path, case_id: int) -> int:
    cases = read_json(out / "cases.json")
    if not 0 <= case_id < len(cases):
        raise ValueError("Case ID outside this experiment")
    case = cases[case_id]
    folder = out / case["path"]
    # Refuse to overwrite or run a case twice, even from competing workers.
    try:
        with (folder / ".started").open("x") as claim:
            claim.write(now())
    except FileExistsError as exc:
        raise DuplicateRun(f"Case already started: {case['path']}") from exc
    random.seed(case["verifier_seed"])
    record = Recorder(folder, case)
    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        record.log(
            f"Task: {case['task']['name']}; strategy: {case['strategy']['name']}; repeat: {case['repeat']}; budget: {budget(case)}"
        )
        record.log(json.dumps(case, indent=2))
        record.save()
        if case["strategy"]["method"] in ("derivation", "astar"):
            deterministic(record)
        else:
            agentic(record)
    except Interrupted as exc:
        record.data.update(status="interrupted", error=str(exc))
    except Exception as exc:
        record.data.update(
            status="error", error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        record.data["finished_utc"] = now()
        record.log(
            f"Finished: {record.data['status']}; {len(record.data['classes'])} classes; {record.data['error'] or 'no error'}"
        )
        record.save()
        render(out)
    return 1 if record.data["status"] in ("error", "interrupted") else 0
