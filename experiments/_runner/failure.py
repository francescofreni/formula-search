"""Reconcile abnormal worker exits using only Python's standard library."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .report import render
from .storage import atomic_text, now, read_json, write_json

# Reserved worker exit: this attempt never acquired ownership of the case.
DUPLICATE_EXIT = 75


def record_failure(
    out: Path, case_id: int, exit_code: int | None, detail: str = ""
) -> None:
    cases = read_json(out / "cases.json")
    if not 0 <= case_id < len(cases):
        raise ValueError("Case ID outside this experiment")
    folder = out / cases[case_id]["path"]
    if exit_code == DUPLICATE_EXIT and (folder / ".started").exists():
        return
    data = read_json(folder / "result.json")
    message = (
        f"Worker exited with code {exit_code}"
        if exit_code is not None
        else "Worker could not start"
    )
    message += f": {detail}" if detail else "; see console.log or Slurm logs"
    data["worker_exit_code"] = exit_code
    data["worker_error"] = message
    # Keep a more specific error and all previously saved discoveries.
    if data["status"] not in ("error", "interrupted"):
        data.update(status="error", error=message)
    data.setdefault("finished_utc", now())
    write_json(folder / "result.json", data)
    path = folder / "README.md"
    text = re.sub(
        r"Status: \*\*[^*]+\*\*",
        f"Status: **{data['status']}**",
        path.read_text(),
        count=1,
    )
    atomic_text(path, text + f"\nWorker failure: {message}\n")
    with (folder / "run.log").open("a") as log:
        log.write(message + "\n")
    render(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(sys.argv[2:])
    record_failure(args.folder.resolve(), args.case, args.exit_code)
