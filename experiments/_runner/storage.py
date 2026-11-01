"""Atomic result files and source snapshots used by local and Slurm workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, data: Any) -> None:
    atomic_text(path, json.dumps(data, indent=2, default=encode) + "\n")


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def snapshot(out: Path) -> dict[str, Any]:
    import hiprof

    import formula_search

    files: dict[str, Path] = {}
    for name, module in (
        ("formula_search", formula_search),
        ("hiprof", hiprof),
    ):
        assert module.__file__
        base = Path(module.__file__).resolve().parent
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".lark"):
                files[f"src/{name}/{p.relative_to(base)}"] = p
    for p in sorted((ROOT / "experiments/_runner").glob("*.py")):
        files[f"experiments/_runner/{p.name}"] = p
    for name in (
        "experiments/run.py",
        "experiments/worker.slurm",
        "experiments/README.md",
        "pyproject.toml",
    ):
        files[name] = ROOT / name
    for p in sorted((ROOT / "experiments/configs").glob("*.json")):
        files[f"experiments/configs/{p.name}"] = p
    hashes = {}
    with tarfile.open(out / "sources.tar.gz", "w:gz") as tar:
        for name, path in sorted(files.items()):
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
            frozen = out / "source" / name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(content)
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    return {"sha256": digest, "files": hashes}


def source_provenance() -> dict[str, Any]:
    """Carry the saved source's identity through subsequent replays."""
    parent_path = ROOT.parent / "manifest.json"
    if ROOT.name == "source" and parent_path.is_file():
        parent = read_json(parent_path)
        return {
            "git_commit": parent["git_commit"],
            "source_parent": {
                "run_id": parent["run_id"],
                "path": str(ROOT.parent),
                "source_sha256": parent["source"]["sha256"],
            },
        }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return {"git_commit": git.stdout.strip(), "source_parent": None}


def prepare(
    spec: dict[str, Any], cases: list[dict[str, Any]], output: Path
) -> Path:
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%d_%H%M%SZ}__{spec['name']}__{uuid4().hex[:6]}"
    out = output.resolve() / run_id
    out.mkdir(parents=True)
    (out / "slurm").mkdir()
    write_json(out / "experiment.json", spec)
    write_json(out / "cases.json", cases)
    source = snapshot(out)
    provenance = source_provenance()
    manifest = {
        "schema": "formula-experiment-v1",
        "run_id": run_id,
        "created_utc": now(),
        "name": spec["name"],
        "python": sys.executable,
        "python_version": platform.python_version(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
        **provenance,
        "source": source,
        "plan_hashes": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("experiment.json", "cases.json")
        },
        "invocation": sys.argv,
        "worker_environment": {
            k: worker_environment()[k]
            for k in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "environment": {
            k: os.environ[k]
            for k in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
            if k in os.environ
        },
    }
    write_json(out / "manifest.json", manifest)
    for case in cases:
        folder = out / case["path"]
        folder.mkdir(parents=True)
        write_json(folder / "config.json", case)
        (folder / "README.md").write_text(
            f"# {case['path']}\n\nStatus: **pending**\n\n"
            "[Complete settings](config.json) · [Results](result.json)\n"
        )
        write_json(
            folder / "result.json",
            {
                "schema": "formula-case-v1",
                "status": "pending",
                "classes": [],
                "discoveries": [],
                "completed_actions": 0,
                "elapsed_seconds": 0,
                "error": None,
                "counts_available": False,
            },
        )
    (out / "REPLAY.md").write_text(
        "# Replay\n\nWorkers use the saved source/ tree, not the current checkout. "
        "sources.tar.gz contains the same code and HiProF grammar.\n\n"
        "Create a Python environment with manifest.json's package versions. "
        "For an archived bundle without source/, extract sources.tar.gz into source/. "
        "Supply API credentials separately. Hosted responses are not exactly reproducible.\n\n"
        "To rerun the saved cases into a NEW result folder:\n\n```bash\n"
        + shlex.join(
            [
                sys.executable,
                str(out / "source/experiments/run.py"),
                "replay",
                str(out),
                "--output-root",
                str(out.parent),
            ]
        )
        + "\n```\n\nThe original result files are never overwritten.\n"
    )
    return out


def worker_command(out: Path, case_id: int) -> list[str]:
    return [
        read_json(out / "manifest.json")["python"],
        str(out / "source/experiments/run.py"),
        "worker",
        str(out),
        "--case",
        str(case_id),
    ]


def worker_environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def verify_source(out: Path) -> None:
    """Reject accidental edits to the recorded source snapshot."""
    manifest = read_json(out / "manifest.json")
    for name, expected in manifest["plan_hashes"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved plan has changed: {name}")
    for name, expected in manifest["source"]["files"].items():
        actual = hashlib.sha256(
            (out / "source" / name).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Saved source has changed: {name}")
