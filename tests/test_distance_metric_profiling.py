"""Repeated estimates retain settings, vary seeds, and include every preset."""

from dataclasses import replace

import pytest

from distance_metric import DistanceResult
from distance_metric_profiling import Config, configs, fleet
from distance_metric_profiling import main as runner


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_bad_repeat_counts(n):
    with pytest.raises(ValueError, match="positive integer"):
        replace(configs.CONFIG, n=n)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_bad_expected_distances(value):
    with pytest.raises(ValueError, match="expected_distance"):
        replace(configs.CONFIG, expected_distance=value)


def test_repeats_keep_settings_vary_seeds_and_summarize(monkeypatch, capsys):
    original = configs.ADJUSTMENT_02_HALF
    config = replace(
        original, n=3, distance=replace(original.distance, seed=42)
    )
    seen = []
    distances = [0.48, 0.52, 0.6]

    def estimate(distance):
        i = len(seen)
        assert distance == replace(config.distance, seed=42 + i)
        seen.append(distance)
        return DistanceResult(
            distance=distances[i],
            interval=(distances[i] - 0.05, distances[i] + 0.05),
            n_chains=distance.n_chains,
            verifier_calls=100 + i,
            verifier_call_budget=200,
            elapsed_seconds=(i + 1) / 10,
            sampling_failure_probability=0.01,
            verifier_failure_bound=2e-12,
            total_failure_bound=0.01 + 2e-12,
        )

    monkeypatch.setattr(runner, "estimate_distance", estimate)
    results = runner.main(config)
    output = capsys.readouterr().out
    assert len(results) == 3
    assert config.distance.seed == 42
    assert "Completed: 3/3" in output
    assert "Mean time: 0.200 s; median: 0.200 s" in output
    assert "Mean distance: 0.533333" in output
    assert "Mean HiProF calls: 101.0" in output
    assert "Within epsilon: 2/3" in output


def test_fleet_runs_each_named_preset_once(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "main", seen.append)
    fleet.main()
    expected = [
        config
        for name, config in vars(configs).items()
        if name != "CONFIG" and isinstance(config, Config)
    ]
    assert len(expected) == 8
    assert seen == expected
    assert all(config.n == 5 for config in seen)
    assert all(config.distance.epsilon == 0.05 for config in seen)
    assert all(config.distance.delta == 0.01 for config in seen)
    controls = [configs.ADJUSTMENT_02_HALF, configs.ADJUSTMENT_03_ONE_THIRD]
    assert all(
        config.distance.optional_edges is None
        and config.distance.topological_order is None
        for config in seen
        if config not in controls
    )
    assert configs.CONFIG.distance.optional_edges is None
    assert configs.CONFIG.distance.topological_order is None


def test_errors_are_not_silently_dropped(monkeypatch, capsys):
    def fail(config):
        raise ValueError("invalid starting formula")

    monkeypatch.setattr(runner, "estimate_distance", fail)
    with pytest.raises(ValueError, match="invalid starting formula"):
        runner.main(configs.CONFIG)
    assert "Mean time:" not in capsys.readouterr().out
