"""Offline checks of experiment isolation, provenance, budgets, and failures."""

import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
cli = importlib.import_module("_runner.cli")
config = importlib.import_module("_runner.config")
storage = importlib.import_module("_runner.storage")
execution = importlib.import_module("_runner.execution")
report = importlib.import_module("_runner.report")


def definition(tmp_path, strategies):
    path = tmp_path / "definition.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "tasks": [
                    {
                        "name": "simple",
                        "graph": "Z -> X; X -> Y",
                        "treatment": "X",
                        "outcome": "Y",
                    }
                ],
                "strategies": strategies,
            }
        )
    )
    return config.load(path)


def prepared(tmp_path, strategies):
    spec, cases = definition(tmp_path, strategies)
    out = storage.prepare(spec, cases, tmp_path / "results")
    report.render(out)
    return out


def result(out, index=0):
    case = storage.read_json(out / "cases.json")[index]
    folder = out / case["path"]
    return folder, storage.read_json(folder / "result.json")


def choice(prompt):
    state = json.loads(prompt)
    options = sorted(
        state["options"], key=lambda m: m["rule"] not in ("base", "submit")
    )
    if "selection_count" in state:
        return json.dumps(
            {"keep": [m["id"] for m in options[: state["selection_count"]]]}
        )
    return json.dumps({"move": options[0]["id"]})


@pytest.mark.parametrize(
    "name,count",
    [
        ("single", 1),
        ("comparison", 32),
        ("agentic", 24),
        ("prompt-comparison", 24),
        ("astar-fleet", 96),
    ],
)
def test_catalog(name, count):
    _, cases = config.load(EXPERIMENTS / "configs" / f"{name}.json")
    assert len(cases) == count
    assert len({c["path"] for c in cases}) == count


def test_filter_resolves_one_case_and_renumbers():
    args = cli.parser().parse_args(
        [
            "plan",
            "comparison",
            "--task",
            "napkin",
            "--strategy",
            "beam",
            "--repeat",
            "2",
        ]
    )
    _, cases = cli.selected(args)
    assert len(cases) == 1
    assert cases[0]["id"] == 0
    assert cases[0]["path"] == "napkin/beam/repeat-02"


@pytest.mark.parametrize(
    "strategy",
    [
        {"method": "beam", "bound": 3},
        {"method": "astar", "max_pops": 0},
        {"method": "depth-first", "max_steps": True},
        {"method": "astar", "name": "../outside"},
        {"method": "depth-first", "prompt_version": "missing"},
    ],
)
def test_invalid_settings_rejected(tmp_path, strategy):
    with pytest.raises(ValueError):
        definition(tmp_path, [strategy])


def test_frozen_workers_and_replay(tmp_path):
    out = prepared(
        tmp_path,
        [
            {"method": "derivation", "bound": 0},
            {"method": "astar", "profile": "uniform", "max_pops": 5},
        ],
    )
    assert all(r["classes"] == "" for r in report.rows(out))
    manifest = storage.read_json(out / "manifest.json")
    assert manifest["worker_environment"]["PYTHONHASHSEED"] == "0"
    assert (out / "source/experiments/configs/comparison.json").is_file()
    assert cli.local_run(out) == 0
    for i in range(2):
        folder, data = result(out, i)
        assert "p(Y | X)" in data["classes"]
        assert data["discoveries"][0]["certified"]
        assert data["discoveries"][0]["derivation"]
        assert "p(Y | X)" in (folder / "README.md").read_text()
    assert result(out, 1)[1]["completed_actions"] <= 5
    # Start from the checkout; replay must delegate to the saved runner.
    replay_root = tmp_path / "replayed"
    assert (
        cli.replay(SimpleNamespace(folder=out, output_root=replay_root)) == 0
    )
    new = next(replay_root.iterdir())
    assert new != out
    assert result(new)[1]["classes"] == result(out)[1]["classes"]
    assert (
        storage.read_json(new / "manifest.json")["source"]
        == manifest["source"]
    )
    with pytest.raises(FileExistsError):
        execution.execute(out, 0)
    (out / "source/experiments/run.py").write_text("# edited")
    with pytest.raises(ValueError, match="Saved source has changed"):
        storage.verify_source(out)


@pytest.mark.parametrize("method", ["depth-first", "beam"])
def test_agents_save_exact_decisions_and_obey_budget(
    tmp_path, monkeypatch, method
):
    settings = {"method": method, "repeats": 1}
    settings.update(
        {"width": 1, "max_layers": 2} if method == "beam" else {"max_steps": 2}
    )
    out = prepared(tmp_path, [settings])
    seen = []

    def fake_client(model, **kwargs):
        assert kwargs["timeout"] == 300
        assert kwargs["max_retries"] == 5

        def complete(prompt):
            seen.append(prompt)
            return choice(prompt)

        return complete

    monkeypatch.setattr(execution, "Fireworks", fake_client)
    assert execution.execute(out, 0) == 0
    folder, data = result(out)
    assert data["completed_actions"] == 2
    assert data["classes"] == ["p(Y | X)"]
    entries = [
        json.loads(line)
        for line in (folder / "decisions.jsonl").read_text().splitlines()
    ]
    assert [e["prompt"] for e in entries if "prompt" in e] == seen
    assert (folder / "prompt.txt").read_text() == json.loads(seen[0])[
        "instructions"
    ]
    row = report.rows(out)[0]
    assert row["method"] == method
    assert row["prompt_sha256"]


def test_failure_preserves_discoveries_and_retries(tmp_path, monkeypatch):
    out = prepared(
        tmp_path, [{"method": "depth-first", "repeats": 1, "max_steps": 5}]
    )
    count = 0

    def fake_client(model, **kwargs):
        def complete(prompt):
            nonlocal count
            count += 1
            if count == 3:
                kwargs["on_retry"]({"event": "retry", "reason": "timeout"})
                raise TimeoutError("offline simulated timeout")
            return choice(prompt)

        return complete

    monkeypatch.setattr(execution, "Fireworks", fake_client)
    assert execution.execute(out, 0) == 1
    folder, data = result(out)
    assert data["status"] == "error"
    assert data["classes"] == ["p(Y | X)", "sum_{Z} { p(Z) p(Y | X, Z) }"]
    assert data["completed_actions"] == 4
    assert data["automatic_steps"] == 2
    assert data["model_calls"] == 3
    assert data["api_attempt_retries"] == 1
    assert "timeout" in data["error"]
    assert "timeout" in (folder / "retries.jsonl").read_text()
    assert report.rows(out)[0]["classes"] == 2


def test_failed_batch_is_unavailable_not_zero(tmp_path, monkeypatch):
    out = prepared(tmp_path, [{"method": "derivation", "bound": 0}])

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted search")

    monkeypatch.setattr(execution, "derive", fail)
    assert execution.execute(out, 0) == 1
    assert report.rows(out)[0]["classes"] == ""
    assert result(out)[1]["status"] == "error"


def test_submit_uses_snapshot_and_separate_logs(tmp_path, monkeypatch):
    out = prepared(tmp_path, [{"method": "astar", "max_pops": 2}])
    spec = storage.read_json(out / "experiment.json")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "12345\n", "")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli.submit(
        out, spec, SimpleNamespace(account="test-account", partition=None)
    )
    command = calls[0]
    assert "--array=0-0%2" in command
    assert f"--output={out}/slurm/%A_%a.out" in command
    assert str(out / "source/experiments/worker.slurm") in command
    assert "--account=test-account" in command
    assert storage.read_json(out / "submission.json")["status"] == "submitted"


def test_documented_replay_location_and_source_ancestry(tmp_path):
    out = prepared(tmp_path, [{"method": "derivation", "bound": 0}])
    original = storage.read_json(out / "manifest.json")
    # Give the containing directory a different HEAD after capturing source.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "unrelated checkout",
        ],
        check=True,
    )
    current = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    assert current != original["git_commit"]
    command = shlex.split(
        (out / "REPLAY.md").read_text().split("```bash\n")[1].split("\n```")[0]
    )
    assert command[-2:] == ["--output-root", str(out.parent)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    new = next(p for p in out.parent.iterdir() if p != out)
    manifest = storage.read_json(new / "manifest.json")
    assert manifest["git_commit"] == original["git_commit"]
    assert manifest["source"] == original["source"]
    assert manifest["replay_of"] == str(out)
    assert manifest["source_parent"]["run_id"] == original["run_id"]
    assert (
        manifest["source_parent"]["source_sha256"]
        == original["source"]["sha256"]
    )
    assert not (out / "source/experiments/results").exists()
    # The bare frozen command must also choose the original run's parent.
    subprocess.run(command[:-2], check=True, capture_output=True, text=True)
    assert len(list(out.parent.iterdir())) == 3
    assert not (out / "source/experiments/results").exists()


@pytest.mark.parametrize(
    "status", ["pending", "running", "error", "interrupted"]
)
def test_nonzero_worker_exit_updates_reports_without_losing_results(
    tmp_path, monkeypatch, status
):
    out = prepared(tmp_path, [{"method": "astar", "max_pops": 2}])
    folder, data = result(out)
    if status != "pending":
        data.update(
            status=status,
            classes=["saved formula"],
            discoveries=[{"formula": "saved formula"}],
            completed_actions=1,
            counts_available=True,
        )
    if status in ("error", "interrupted"):
        data["error"] = "specific search error"
    storage.write_json(folder / "result.json", data)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 42),
    )
    assert cli.local_run(out) == 1
    _, saved = result(out)
    assert saved["status"] == (
        "interrupted" if status == "interrupted" else "error"
    )
    assert saved["worker_exit_code"] == 42
    for field in (
        "classes",
        "discoveries",
        "completed_actions",
        "counts_available",
    ):
        assert saved[field] == data[field]
    if status in ("error", "interrupted"):
        assert saved["error"] == "specific search error"
    assert report.rows(out)[0]["status"] == saved["status"]
    assert (
        f"Status: **{saved['status']}**" in (folder / "README.md").read_text()
    )
    assert "42" in (out / "summary.csv").read_text() or status in (
        "error",
        "interrupted",
    )


def test_worker_launch_exception_is_reported(tmp_path, monkeypatch):
    out = prepared(tmp_path, [{"method": "derivation", "bound": 0}])

    def missing(*args, **kwargs):
        raise FileNotFoundError("missing interpreter")

    monkeypatch.setattr(cli.subprocess, "run", missing)
    assert cli.local_run(out) == 1
    _, data = result(out)
    assert data["status"] == "error"
    assert "missing interpreter" in data["error"]
    assert report.rows(out)[0]["classes"] == ""


@pytest.mark.parametrize("module_code,srun_code", [(23, 0), (0, 42)])
def test_slurm_bootstrap_failures_reach_reports(
    tmp_path, module_code, srun_code
):
    out = prepared(tmp_path, [{"method": "derivation", "bound": 0}])
    startup = tmp_path / "bash_env"
    startup.write_text(
        f"module() {{ return {module_code}; }}\nsrun() {{ return {srun_code}; }}\n"
    )
    env = {**os.environ, "BASH_ENV": str(startup), "SLURM_ARRAY_TASK_ID": "0"}
    process = subprocess.run(
        [
            "bash",
            str(out / "source/experiments/worker.slurm"),
            str(out),
            sys.executable,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == (module_code or srun_code)
    _, data = result(out)
    assert data["status"] == "error"
    assert data["worker_exit_code"] == (module_code or srun_code)
    assert "error" in (out / "README.md").read_text()
    # The EXIT trap ran its reporter with -S: search dependencies unavailable.
    assert "Could not record worker failure" not in process.stderr


@pytest.mark.parametrize("status", ["exhausted", "running"])
def test_duplicate_slurm_worker_preserves_owner_artifacts(tmp_path, status):
    out = prepared(tmp_path, [{"method": "derivation", "bound": 0}])
    assert execution.execute(out, 0) == 0
    folder, data = result(out)
    data["status"] = status
    storage.write_json(folder / "result.json", data)
    report.render(out)
    artifacts = [p for p in folder.iterdir() if p.is_file()]
    artifacts += [out / "README.md", out / "summary.csv"]
    before = {p: p.read_bytes() for p in artifacts}
    startup = tmp_path / "bash_env"
    startup.write_text('module() { :; }\nsrun() { "$@"; }\n')
    process = subprocess.run(
        [
            "bash",
            str(out / "source/experiments/worker.slurm"),
            str(out),
            sys.executable,
        ],
        env={
            **os.environ,
            "BASH_ENV": str(startup),
            "SLURM_ARRAY_TASK_ID": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 75
    assert "Case already started" in process.stderr
    assert {p: p.read_bytes() for p in artifacts} == before


def test_recovery_logged_with_actual_calls_and_forced_steps(
    tmp_path, monkeypatch
):
    out = prepared(
        tmp_path, [{"method": "depth-first", "repeats": 1, "max_steps": 3}]
    )
    calls = 0

    def fake_client(model, **kwargs):
        def complete(prompt):
            nonlocal calls
            calls += 1
            return '{"move":-1}' if calls == 2 else choice(prompt)

        return complete

    monkeypatch.setattr(execution, "Fireworks", fake_client)
    assert execution.execute(out, 0) == 0
    folder, data = result(out)
    assert data["completed_actions"] == 3
    assert data["model_calls"] == 3
    assert data["decision_retries"] == 1
    assert data["automatic_steps"] == 1
    assert data["api_attempt_retries"] == 0
    events = [
        json.loads(s)
        for s in (folder / "decisions.jsonl").read_text().splitlines()
    ]
    requests = [e for e in events if e.get("event") == "request"]
    assert [(e["step"], e["attempt"]) for e in requests] == [
        (1, 1),
        (3, 1),
        (3, 2),
    ]
    assert "correction" in json.loads(requests[-1]["prompt"])
    assert any(
        e.get("event") == "automatic" and e["step"] == 2 for e in events
    )
    assert (
        '"event": "decision_retry"' in (folder / "retries.jsonl").read_text()
    )
    row = report.rows(out)[0]
    assert row["model_calls"] == 3
    assert row["automatic_steps"] == 1
    assert row["decision_retries"] == 1
    assert row["prompt_layout"] == "depth-first-v1"


def test_beam_forced_layer_is_logged_without_model_calls(
    tmp_path, monkeypatch
):
    out = prepared(
        tmp_path,
        [{"method": "beam", "width": 100, "max_layers": 1, "repeats": 1}],
    )

    def fake_client(model, **kwargs):
        return lambda _: pytest.fail("No model call needed for this layer")

    monkeypatch.setattr(execution, "Fireworks", fake_client)
    assert execution.execute(out, 0) == 0
    folder, data = result(out)
    assert data["completed_actions"] == 1
    assert data["automatic_steps"] == 1
    assert data["model_calls"] == 0
    assert data["status"] == "step_limit"
    assert '"event": "automatic"' in (folder / "decisions.jsonl").read_text()
    assert report.rows(out)[0]["prompt_layout"] == "beam-parents-v1"
