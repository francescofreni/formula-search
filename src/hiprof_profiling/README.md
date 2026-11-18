# HiProF timing

Select `CONFIG` at the bottom of `configs.py`, then run from the repository root:

```bash
.venv/bin/python -m hiprof_profiling
```

With the virtual environment activated, directory execution also works:

```bash
# From src/
python hiprof_profiling/

# From src/hiprof_profiling/
python .
```

`target` is the interventional equation to compare with `formula`; the
observational formula alone does not specify the causal effect to check.
`n` sets the number of checks. Each timed call creates a fresh verifier and
checks the equation. Timings include graph construction, parsing, and
verification, but exclude imports and printing. There is no warm-up; the
first call is included and each call's time is printed separately.

The seed is set once, so successive checks draw fresh samples from a
reproducible sequence. `target_bound` controls HiProF's false-acceptance
bound within its linear-Gaussian model family. An accepted result is not a
general identification proof. Malformed inputs raise HiProF's exception.

You can also pass a config directly:

```python
from hiprof_profiling import Config
from hiprof_profiling.main import main

main(Config(
    graph="X -> Y",
    formula="p(Y | X)",
    target="p(Y | do(X))",
    n=10,
))
```

## Named presets

`configs.py` contains 13 formulas copied exactly from saved full-search,
A*, and agentic beam-search results, grouped by graph:

| Group | Presets | Examples |
| --- | ---: | --- |
| `ADJUSTMENT` | 5 | Adjust for A; nested adjustment; conditional division; nested averaging |
| `MEDIATION` | 5 | Adjust for Z; mediator factorization; separate adjustments; nested conditional divisions |
| `FRONT_DOOR` | 1 | Standard front-door formula |
| `NAPKIN` | 2 | Conditional division with marginal or conditional weights |

Names follow `GRAPH_NN_DESCRIPTION`, with a zero-padded rank starting at 01.
Order is by probability-term count, then number of sum/icd/quotient operators;
equal-complexity entries are allowed. This does not predict runtime.
No artificial quotient expansions are added. See [SOURCES.md](SOURCES.md)
for exact saved files, formula indices, complexity counts, and source hashes.

The adjustment group uses the search graph
`A -> B; B -> X; A -> Y; X -> Y`, rather than the earlier three-node example.
Front-door uses treatment `T`, matching its saved results. Each group's 01
config supplies the graph, target, and defaults; other presets inherit them
using `dataclasses.replace`.

Select and customize a preset at the bottom of `configs.py`:

```python
CONFIG = replace(MEDIATION_05_NESTED_CONDITIONAL_DIVISIONS, n=100)
```

The `Config` dataclass remains in `config.py`; the timing loop is in
`main.py`. Both module and directory execution use the same selected preset.

## Run the whole fleet

From the repository root:

```bash
.venv/bin/python -m hiprof_profiling.fleet
```

Or, with the virtual environment activated, from `src/hiprof_profiling/`:

```bash
python fleet.py
```

The fleet runs every named `Config` in `configs.py` sequentially, in file
order, using each preset's `n`, seed, and tolerance. New presets are picked
up automatically. The single-run `CONFIG` alias is excluded so it is not
run twice. Each block prints the preset name, settings, accepted/trial
count, mean time, and median time. An exception stops the fleet at the
failing config, as with the single-config script.
