"""One interface for planning, local execution, Slurm, and past results."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import config
from .budgets import budget
from .execution import DuplicateRun, execute
from .failure import DUPLICATE_EXIT, record_failure
from .report import history, render
from .storage import (
    ROOT,
    prepare,
    read_json,
    verify_source,
    worker_command,
    worker_environment,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list", help="List named definitions, graphs, and A* profiles"
    )
    for name in ("plan", "prepare", "run", "submit"):
        p = commands.add_parser(name)
        p.add_argument("experiment", help="Definition name or JSON path")
        p.add_argument(
            "--task",
            action="append",
            help="Select a graph by name; repeatable",
        )
        p.add_argument(
            "--strategy",
            action="append",
            help="Select a strategy name or method; repeatable",
        )
        p.add_argument(
            "--repeat", type=int, help="Select one repeat (1-based)"
        )
        p.add_argument("--name", help="Custom label for this cohort")
        p.add_argument(
            "--output-root", type=Path, default=ROOT / "experiments/results"
        )
        if name == "submit":
            p.add_argument("--account")
            p.add_argument("--partition")
    p = commands.add_parser("worker", help=argparse.SUPPRESS)
    p.add_argument("folder", type=Path)
    p.add_argument("--case", type=int, required=True)
    p = commands.add_parser(
        "report", help="Refresh a run's readable table and CSV"
    )
    p.add_argument("folder", type=Path)
    commands.add_parser(
        "history", help="Index saved runs and link legacy result folders"
    )
    p = commands.add_parser(
        "replay", help="Rerun a frozen experiment in a fresh folder"
    )
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--output-root",
        type=Path,
        help="Destination for the new run (default: original run's parent)",
    )
    return root


def selected(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(args.experiment)
    if not path.is_file():
        path = ROOT / "experiments/configs" / f"{args.experiment}.json"
    spec, cases = config.load(path)
    for values, available, label in (
        (args.task, {c["task"]["name"] for c in cases}, "task"),
        (
            args.strategy,
            {c["strategy"]["name"] for c in cases} | set(config.METHODS),
            "strategy",
        ),
    ):
        if values and set(values) - available:
            raise ValueError(
                f"Unknown {label} selection: {set(values) - available}"
            )
    cases = [
        c
        for c in cases
        if (not args.task or c["task"]["name"] in args.task)
        and (
            not args.strategy
            or c["strategy"]["name"] in args.strategy
            or c["strategy"]["method"] in args.strategy
        )
        and (args.repeat is None or c["repeat"] == args.repeat)
    ]
    if not cases:
        raise ValueError("Selection contains no cases")
    cases = [{**c, "id": i} for i, c in enumerate(cases)]
    spec["selection"] = {
        "tasks": args.task,
        "strategies": args.strategy,
        "repeat": args.repeat,
    }
    if args.name:
        spec["name"] = config.slug(args.name)
    return spec, cases


def plan(spec: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [
        f"{spec['name']}: {len(cases)} independent cases",
        spec["description"],
        "",
    ]
    for c in cases:
        s = c["strategy"]
        prompt = (
            f"; prompt={s['prompt_version']}" if "prompt_version" in s else ""
        )
        lines.append(f"{c['id']:3}: {c['path']} — {budget(c)}{prompt}")
    agents = sum(
        c["strategy"]["method"] in ("depth-first", "beam") for c in cases
    )
    lines.append(
        f"\nModel/API cases: {agents}. Slurm concurrency: {spec['slurm']['concurrency']}."
    )
    return "\n".join(lines)


def local_run(out: Path) -> int:
    failures = 0
    for case in read_json(out / "cases.json"):
        print(f"Running {case['path']}", flush=True)
        with (out / case["path"] / "console.log").open("x") as console:
            try:
                result = subprocess.run(
                    worker_command(out, case["id"]),
                    env=worker_environment(),
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError as exc:
                record_failure(out, case["id"], None, str(exc))
                failures += 1
                continue
        if result.returncode:
            record_failure(out, case["id"], result.returncode)
            failures += 1
    render(out)
    print(f"Results: {out / 'README.md'}\nFailed workers: {failures}")
    return int(failures > 0)


def submit(out: Path, spec: dict[str, Any], args: argparse.Namespace) -> None:
    settings = spec["slurm"]
    count = len(read_json(out / "cases.json"))
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=fs-{spec['name']}",
        f"--array=0-{count - 1}%{settings['concurrency']}",
        f"--chdir={out}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        f"--mem={settings['memory']}",
        f"--time={settings['time']}",
        f"--output={out}/slurm/%A_%a.out",
        f"--error={out}/slurm/%A_%a.err",
    ]
    for name in ("account", "partition"):
        if getattr(args, name):
            command.append(f"--{name}={getattr(args, name)}")
    command += [
        str(out / "source/experiments/worker.slurm"),
        str(out),
        sys.executable,
    ]
    write_json(
        out / "submission.json", {"command": command, "status": "submitting"}
    )
    (out / "submission.txt").write_text(shlex.join(command) + "\n")
    result = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    write_json(
        out / "submission.json",
        {
            "command": command,
            "status": "submitted" if result.returncode == 0 else "failed",
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        raise RuntimeError(
            f"Slurm rejected submission: {result.stderr.strip()}"
        )
    print(f"Submitted job {result.stdout.strip()}\nResults: {out}")


def replay(args: argparse.Namespace) -> int:
    old = args.folder.resolve()
    verify_source(old)
    output = (args.output_root or old.parent).resolve()
    # Delegate to the saved code before constructing a new run.
    if ROOT != old / "source":
        command = [
            sys.executable,
            str(old / "source/experiments/run.py"),
            "replay",
            str(old),
            "--output-root",
            str(output),
        ]
        return subprocess.run(command, check=False).returncode
    spec, cases = read_json(old / "experiment.json"), read_json(
        old / "cases.json"
    )
    spec["name"] += "-replay"
    out = prepare(spec, cases, output)
    manifest = read_json(out / "manifest.json")
    manifest["replay_of"] = str(old)
    write_json(out / "manifest.json", manifest)
    render(out)
    return local_run(out)


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "list":
        for p in sorted((ROOT / "experiments/configs").glob("*.json")):
            spec, cases = config.load(p)
            print(f"{p.stem}: {len(cases)} cases — {spec['description']}")
        print("\nGraphs:", ", ".join(config.TASKS))
        print("A* profiles:", ", ".join(config.PROFILES_BY_NAME))
        return 0
    if args.command == "worker":
        out = args.folder.resolve()
        verify_source(out)
        if ROOT != out / "source":
            return subprocess.run(
                worker_command(out, args.case),
                env=worker_environment(),
                check=False,
            ).returncode
        return execute(out, args.case)
    if args.command == "report":
        print(render(args.folder.resolve()))
        return 0
    if args.command == "history":
        print(history(ROOT / "experiments"))
        return 0
    if args.command == "replay":
        return replay(args)
    spec, cases = selected(args)
    print(plan(spec, cases))
    if args.command == "plan":
        return 0
    out = prepare(spec, cases, args.output_root)
    render(out)
    print(f"Prepared: {out}")
    if args.command == "submit":
        submit(out, spec, args)
        return 0
    return local_run(out) if args.command == "run" else 0


def main() -> None:
    try:
        code = dispatch(parser().parse_args())
    except DuplicateRun as exc:
        print(str(exc), file=sys.stderr)
        code = DUPLICATE_EXIT
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(code)
