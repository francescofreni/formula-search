"""Submit or execute matched runtime trials using a frozen source snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any
from uuid import uuid4

from distance_metric import DistanceConfig, estimate_distance
from formula_search.evaluator import Archive, Task

GRAPH = "A -> X; A -> Y; B -> X; B -> Y; X -> M; M -> N; N -> Y"
FORMULAS = (
    "sum_{A, B} { p(A, B) p(Y | X, A, B) }",
    "sum_{A, B} { p(A) p(B) p(Y | X, A, B) }",
    "sum_{A, B, M} { p(A, B) p(M | X) p(Y | M, A, B) }",
    "sum_{A, B, N} { p(A, B) p(N | X) p(Y | N, A, B) }",
    "sum_{A, B, M, N} { p(A, B) p(M | X) p(N | M) p(Y | N, A, B) }",
)
BASE = DistanceConfig(graph=GRAPH, target="p(Y | do(X))", formulas=FORMULAS)
CASES = {
    "two_formulas": replace(BASE, formulas=(FORMULAS[0], FORMULAS[4])),
    "five_formulas": BASE,
}
COLUMNS = (
    "case",
    "trial",
    "seed",
    "order_in_pair",
    "n_formulas",
    "n_pairs",
    "elapsed_seconds",
    "distance",
    "n_chains",
    "verifier_calls",
    "verifier_call_budget",
    "topological_permutation_attempts",
    "host",
    "slurm_job_id",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(trials: int, parent: Path) -> Path:
    if trials < 1:
        raise ValueError("trials must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (parent / f"{stamp}_group_runtime_{uuid4().hex[:6]}").resolve()
    root.mkdir(parents=True)
    (root / "slurm").mkdir()
    # Save the actual installed implementation, including uncommitted edits
    # and HiProF's grammar. Workers import only these saved packages.
    for name in ("distance_metric", "formula_search", "hiprof"):
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve().parent
        for path in source.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".lark"):
                target = root / "source" / name / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    for name in ("profile.py", "run.slurm"):
        shutil.copyfile(Path(__file__).with_name(name), root / "source" / name)
    plan = {
        "trials_per_case": trials,
        "seeds": list(range(trials)),
        "cases": {name: asdict(config) for name, config in CASES.items()},
        "schedule": "Sequential pairs; reverse case order on odd seeds.",
        "timing": (
            "elapsed_seconds comes from estimate_distance: includes base "
            "checks, rejection sampling, graph construction, HiProF checks "
            "and pair averaging. Excludes imports, archive preflight and "
            "result writing. No discarded warmups or extra timing wrappers."
        ),
    }
    write_json(root / "config.json", plan)
    write_json(
        root / "manifest.json",
        {
            "created_utc": now(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "submission_host": platform.node(),
            "invocation": sys.argv,
            "config_sha256": digest(root / "config.json"),
            "source_hashes": {
                str(p.relative_to(root / "source")): digest(p)
                for p in sorted((root / "source").rglob("*"))
                if p.is_file()
            },
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
        },
    )
    write_json(
        root / "status.json",
        {
            "status": "prepared",
            "expected_trials": trials * 2,
            "completed_trials": 0,
        },
    )
    (root / "README.md").write_text(
        "# Distance runtime experiment\n\n"
        f"{trials} trials each for 2 and 5 formulas. "
        "Same six-variable graph; the pair uses formulas 1 and 5.\n\n"
        "- `timings.csv`: histogram input; filter `case`, plot "
        "`elapsed_seconds` in seconds.\n"
        "- `trials/*.json`: full result and seed for each completed trial.\n"
        "- `config.json`: exact graph, formulas, settings and schedule.\n"
        "- `status.json`: completion/failure and expected trial count.\n"
        "- `summary.json`: final per-case runtime statistics.\n"
        "- `manifest.json`, `source/`: frozen code, HiProF grammar, "
        "package versions and hashes.\n"
        "- `submission.json`, `slurm/`: exact submitted command and logs.\n\n"
        "Runs are sequential on one allocated CPU. Seeds match across "
        "cases; case order alternates. Trials are included from the first "
        "one onward, with no discarded warmups. An error stops the job; "
        "completed trials remain saved. Only status=complete means all "
        "requested samples are present.\n"
    )
    return root


def verify_snapshot(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(root / "config.json") != manifest["config_sha256"]:
        raise ValueError("Saved config changed after preparation")
    for name, expected in manifest["source_hashes"].items():
        if digest(root / "source" / name) != expected:
            raise ValueError(f"Saved source changed: {name}")
    for name in ("distance_metric", "formula_search", "hiprof"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(root / "source"):
            raise ValueError(
                f"Worker imported live code instead of snapshot: {path}"
            )
    return json.loads((root / "config.json").read_text())


def worker(root: Path) -> None:
    root = root.resolve()
    # Refuse overwrite or accidental reruns; each submission has its own folder.
    with (root / "worker.lock").open("x"):
        pass
    rows: list[dict[str, Any]] = []
    expected = 0
    try:
        plan = verify_snapshot(root)
        expected = 2 * plan["trials_per_case"]
        configs = {
            name: DistanceConfig(**cfg) for name, cfg in plan["cases"].items()
        }
        write_json(
            root / "status.json",
            {
                "status": "running",
                "started_utc": now(),
                "expected_trials": expected,
                "completed_trials": 0,
                "host": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            },
        )
        random.seed(0)
        archive = Archive(Task(GRAPH, "X", "Y"))
        for formula in configs["five_formulas"].formulas:
            verdict = archive.submit(formula)
            if verdict.outcome != "new":
                raise ValueError(f"Preflight formula rejected: {verdict}")
        archive.save(root / "preflight_archive.json")
        (root / "trials").mkdir()
        with (root / "timings.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            stream.flush()
            for seed in plan["seeds"]:
                schedule = list(configs.items())
                if seed % 2:
                    schedule.reverse()
                matched = None
                for order, (name, config) in enumerate(schedule, 1):
                    print(
                        f"START {name}: trial {seed + 1}/"
                        f"{plan['trials_per_case']}, seed={seed}",
                        flush=True,
                    )
                    result = estimate_distance(replace(config, seed=seed))
                    signature = (
                        result.topological_permutation_attempts,
                        result.topological_order_counts,
                    )
                    if matched is not None and signature != matched:
                        raise ValueError("Paired sampling diagnostics differ")
                    matched = signature
                    details = asdict(result)
                    details["topological_order_counts"] = [
                        {"order": topo, "chains": count}
                        for topo, count in sorted(
                            result.topological_order_counts.items()
                        )
                    ]
                    row = {
                        "case": name,
                        "trial": seed + 1,
                        "seed": seed,
                        "order_in_pair": order,
                        "host": platform.node(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                    }
                    row.update(
                        {
                            key: details[key]
                            for key in COLUMNS
                            if key in details
                        }
                    )
                    write_json(
                        root / "trials" / f"{name}_{seed:02d}.json",
                        {**row, "result": details},
                    )
                    writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    write_json(
                        root / "status.json",
                        {
                            "status": "running",
                            "expected_trials": expected,
                            "completed_trials": len(rows),
                            "updated_utc": now(),
                        },
                    )
                    print(
                        f"DONE {name}: {result.elapsed_seconds:.3f} s; "
                        f"distance={result.distance:.6f}; "
                        f"HiProF calls={result.verifier_calls}",
                        flush=True,
                    )
        summary = {}
        for name in configs:
            times = [
                row["elapsed_seconds"] for row in rows if row["case"] == name
            ]
            if len(times) != plan["trials_per_case"]:
                raise ValueError("Incomplete case")
            summary[name] = {
                "trials": len(times),
                "mean_seconds": mean(times),
                "std_seconds": stdev(times) if len(times) > 1 else None,
                "min_seconds": min(times),
                "max_seconds": max(times),
            }
        write_json(root / "summary.json", summary)
        write_json(
            root / "status.json",
            {
                "status": "complete",
                "expected_trials": expected,
                "completed_trials": len(rows),
                "finished_utc": now(),
            },
        )
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        write_json(
            root / "status.json",
            {
                "status": "failed",
                "expected_trials": expected,
                "completed_trials": len(rows),
                "error": str(error),
                "finished_utc": now(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "worker"))
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    if args.action == "worker":
        if args.root is None:
            parser.error("worker requires a run folder")
        worker(args.root)
        return
    if args.trials < 1:
        parser.error("--trials must be positive")
    root = prepare(args.trials, args.root or Path(__file__).parent / "results")
    print(f"Run folder: {root}", flush=True)
    if args.action == "submit":
        command = [
            "sbatch",
            "--parsable",
            f"--output={root}/slurm/%j.out",
            f"--error={root}/slurm/%j.err",
            str(root / "source/run.slurm"),
            str(root),
            sys.executable,
        ]
        print("Submitting:", shlex.join(command), flush=True)
        write_json(
            root / "submission.json",
            {"command": command, "submitted_utc": now()},
        )
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        job_id = result.stdout.strip().split(";")[0]
        write_json(
            root / "submission.json",
            {
                "command": command,
                "submitted_utc": now(),
                "job_id": job_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        print(f"Submitted job {job_id}", flush=True)


if __name__ == "__main__":
    main()
