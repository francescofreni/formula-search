# Distance-metric profiling

Select `CONFIG` at the bottom of `configs.py`. Run from the repository root:

```bash
.venv/bin/python -m distance_metric_profiling
.venv/bin/python -m distance_metric_profiling.fleet
```

With the virtual environment activated, from `src/distance_metric_profiling/`:

```bash
python .         # Selected config
python fleet.py  # All named configs
```

## Settings

`Config` wraps a `distance_metric.DistanceConfig` and adds `n`, the number
of **complete distance estimates**. Default `n=5` runs seeds 0 through 4.
Each estimate samples **1,060 chains**, using epsilon 0.05 and delta 0.01.
Successive repeats use `distance.seed + i`, so they do not repeat identical
chains. Settings are printed before each benchmark.

```python
# At the bottom of configs.py:
CONFIG = replace(ADJUSTMENT_04_EIGHT_EDGES, n=10)

# Change the inner distance settings separately:
CONFIG = replace(
    CONFIG,
    distance=replace(CONFIG.distance, seed=100),
)
```

The fleet uses each named preset's settings, skipping the `CONFIG` alias.
Changing only `CONFIG` affects single-config execution, not the fleet.
All runs are sequential. No verifier cache is shared between estimates.

## Presets

Names use `GRAPH_NN_DESCRIPTION`, grouped by graph. The rank orders
scenarios; it does not claim increasing runtime. Increasing optional edges
also changes the distance being measured, not just its computation cost.

| Preset | Optional edges | Expected distance |
| --- | ---: | --- |
| `ADJUSTMENT_01_IDENTICAL` | 8 | 0 |
| `ADJUSTMENT_02_HALF` | 2 | 1/2 |
| `ADJUSTMENT_03_ONE_THIRD` | 3 | 1/3 |
| `ADJUSTMENT_04_EIGHT_EDGES` | 8 | Not established |
| `MEDIATION_01_ADJUSTMENT_VS_FACTORIZATION` | 8 | Not established |
| `MEDIATION_02_ADJUSTMENT_VS_NESTED_DIVISIONS` | 8 | Not established |
| `FRONT_DOOR_01_IDENTICAL` | 3 | 0 |
| `NAPKIN_01_MARGINAL_VS_CONDITIONAL_WEIGHTS` | 7 | Not established |

The mediation, front-door, and napkin formulas reuse the actual search
outputs in [hiprof_profiling/configs.py](../hiprof_profiling/configs.py).
Their provenance is recorded in [SOURCES.md](../hiprof_profiling/SOURCES.md).
The adjustment cases reuse the known-distance examples in
[distance_metric/configs.py](../distance_metric/configs.py).

## Output and interpretation

Each repeat prints its seed, estimate, elapsed seconds, and HiProF calls.
Automatic experiments also report permutation attempts, rejections, and
accepted-order counts with their generated edges.
The summary prints completed runs, mean/median time, mean/range of distance,
and mean verifier calls. Where the exact expectation is known, it also
reports how many estimates were within epsilon. Zero distance is a valid
result, not a failed run. Exceptions stop execution with a traceback; failed
estimates are not silently excluded from a purported completed benchmark.

Timings use the estimator's elapsed time: validation, graph construction,
chain sampling, and verification, excluding imports and reporting. There is
no unreported warm-up; the first estimate is included. `main(config)` also
returns the individual `DistanceResult` objects for further analysis.

All guarantees are for HiProF's linear-Gaussian model family. Delta 0.01 is
per estimate, separately from verifier error; it is not a simultaneous 99%
guarantee for the whole fleet. The arithmetic mean of repeat estimates is a
descriptive summary; this script does not assign it a new confidence bound.
See [the estimator README](../distance_metric/README.md) for exact semantics.

The default `ADJUSTMENT_04_EIGHT_EDGES` and all other presets except the
known-distance `ADJUSTMENT_02_HALF` and `ADJUSTMENT_03_ONE_THIRD` controls
now generate optional edges automatically. Each chain samples a fresh
uniform permutation of all variables, rejects backward-pointing arrows,
and retries until valid. It then generates and shuffles the allowed edges.
These particular benchmark graphs have unique valid topological orders;
the algorithm still performs rejection sampling on every chain.

Set the nested distance config's `include_bidirected=False` for directed
additions only. Do not supply `optional_edges` or `topological_order` for
the normal automatic experiment. Explicit edge lists or an explicit order
select a different, fixed-family distribution and are labeled in the logs.
See [edge generation](../distance_metric/README.md#automatic-optional-edges)
for the precise distribution and the cost of rejection sampling.
