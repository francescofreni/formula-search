"""Histogram trials preserve pairing, counts, saved rows and failures."""

import csv
import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import stdev

import pytest

from distance_metric import DistanceResult

PATH = (
    Path(__file__).resolve().parents[1]
    / "distance_metric_profiling/profile.py"
)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("distance_runtime", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(config):
    return DistanceResult(
        distance=0.5,
        interval=(0.45, 0.55),
        n_chains=config.n_chains,
        verifier_calls=10,
        verifier_call_budget=20,
        elapsed_seconds=float(len(config.formulas)),
        sampling_failure_probability=config.delta,
        verifier_failure_bound=1e-12,
        total_failure_bound=0.010000000001,
        n_formulas=len(config.formulas),
        n_pairs=config.n_pairs,
        sampling_mode="uniform_topological_orders",
        topological_permutation_attempts=2000 + config.seed,
        topological_order_counts={
            ("A", "B", "X", "M", "N", "Y"): config.n_chains
        },
    )


def local_plan(runner, monkeypatch):
    # Integration smoke checks actual frozen imports separately. Unit tests
    # inject the plan so the worker loop can run in this pytest interpreter.
    monkeypatch.setattr(
        runner,
        "verify_snapshot",
        lambda root: json.loads((root / "config.json").read_text()),
    )


def test_thirty_per_case_balanced_order_and_histogram_rows(
    runner, monkeypatch, tmp_path, capsys
):
    root = runner.prepare(30, tmp_path)
    local_plan(runner, monkeypatch)
    calls = []

    def estimate(config):
        calls.append((len(config.formulas), config.seed))
        assert config.n_chains == 1060
        assert config.optional_edges is config.topological_order is None
        assert config.include_bidirected
        distance = 0.2 + config.seed / 100
        return replace(
            fake_result(config),
            distance=distance,
            interval=(distance - 0.05, distance + 0.05),
        )

    monkeypatch.setattr(runner, "estimate_distance", estimate)
    runner.worker(root)
    expected = [
        (n, seed)
        for seed in range(30)
        for n in ((2, 5) if seed % 2 == 0 else (5, 2))
    ]
    assert calls == expected
    with (root / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60
    assert Counter(row["case"] for row in rows) == {
        "two_formulas": 30,
        "five_formulas": 30,
    }
    assert Counter(
        row["case"] for row in rows if row["order_in_pair"] == "1"
    ) == {"two_formulas": 15, "five_formulas": 15}
    assert len(list((root / "trials").glob("*.json"))) == 60
    assert all(
        float(row["elapsed_seconds"]) == int(row["n_formulas"]) for row in rows
    )
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["completed_trials"] == status["expected_trials"] == 60
    summary = json.loads((root / "summary.json").read_text())
    for case in summary.values():
        assert case["mean_distance"] == pytest.approx(0.345)
        assert case["min_distance"] == pytest.approx(0.2)
        assert case["max_distance"] == pytest.approx(0.49)
        assert case["std_distance"] == pytest.approx(
            stdev(0.2 + seed / 100 for seed in range(30))
        )
    output = capsys.readouterr().out
    assert "distance=0.200000; interval=[0.150000, 0.250000]" in output
    assert '"mean_distance"' in output
    before = (root / "timings.csv").read_bytes()
    with pytest.raises(FileExistsError):
        runner.worker(root)
    assert (root / "timings.csv").read_bytes() == before


def test_failure_preserves_completed_trials_and_does_not_claim_completion(
    runner, monkeypatch, tmp_path
):
    root = runner.prepare(1, tmp_path)
    local_plan(runner, monkeypatch)

    def estimate(config):
        if len(config.formulas) == 5:
            raise RuntimeError("test failure")
        return fake_result(config)

    monkeypatch.setattr(runner, "estimate_distance", estimate)
    with pytest.raises(RuntimeError, match="test failure"):
        runner.worker(root)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_trials"] == 1
    assert status["expected_trials"] == 2
    with (root / "timings.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    assert not (root / "summary.json").exists()


def test_snapshot_checks_config_hash_and_import_location(runner, tmp_path):
    root = runner.prepare(1, tmp_path)
    with pytest.raises(ValueError, match="live code"):
        runner.verify_snapshot(root)
    with (root / "config.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Saved config changed"):
        runner.verify_snapshot(root)
