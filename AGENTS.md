# Repository guidance

## Purpose and architecture

This Python library searches for distinct causal identifying formulas in
HiProF's grammar. Read `README.md` for usage and `PROGRESS.md` for the
implementation state and search design. The broader plan is maintained in
the sibling `formula-search-docs` repository when available.

- `graph.py`: ADMG parsing, graph operations, m-separation, reachable
  closures, and tests for equality constraints.
- `expression.py`: immutable kernel expressions (`Term`, `Product`, `Sum`,
  `ICD`), simplification, and rendering with bound-variable renaming.
- `docalculus.py`: Rules 2 and 3 and breadth-first traversal of equivalent
  queries, retaining shortest paths.
- `baselines/`: ID/IDC, IDC on equivalent queries, and enumeration of
  adjustment, front-door, front-door with adjustment, and napkin templates.
- `derivation.py`: memoized AND/OR search ordered by the number of
  probability rules; do-calculus steps are free. Returns replayable trees.
- `evaluator.py`: validates submissions, archives distinct formula classes,
  and records single-edge additions that invalidate each new class.
- `suite.py`: curated examples and seeded random identifiable tasks.

## Semantic requirements

- Expressions are kernels with explicit output and input sets. Preserve
  admissibility, sequential products, and capture-free variable rendering.
- Simplification identities must hold for every positive observational law;
  graph-specific equivalences belong in causal reasoning and verification.
- Formula validity is checked in the task graph. Formula classes are
  compared on a complete DAG, whose observed model is unconstrained.
  Equality only in the task graph does not make two formulas duplicates.
- HiProF acceptance is verification support relative to its linear Gaussian
  checks, not a general derivation certificate. Certificates require a
  replayable `Derivation`; `Archive.submit` verifies supplied derivations.
- Search uses HiProF for deduplication; validity comes from its rules.
  An empty bounded search result does not establish non-identifiability.
- Preserve deterministic enumeration and explicit random seeds.
- Search and template enumeration can grow rapidly. Use small graphs and
  bounds for exploratory runs.

## Environment and checks

Requires Python 3.11+ and **the `hiprof-eq` version of `hiprof`**, rather than
the older PyPI interface. Installation from a sibling checkout:

```bash
pip install -e ../hiprof-eq
pip install -e '.[dev]'
```

An existing `.venv` can be used directly. Development checks:

```bash
.venv/bin/pytest
.venv/bin/black --check src tests
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

Tests mirror the module layout and include known causal examples,
randomized checks, grammar validation, and archive class counts. For
semantic changes, run the relevant tests and the full suite when feasible.
Follow the existing typed Python style and 79-character formatting.

## Current scope

Archives are in memory. The documented remaining work includes experiment
orchestration and persistence, baseline archives, bottom-up enumeration,
search diagnostics/frontiers, and managing formula-class growth. Consult
the code and `PROGRESS.md` before treating a planned feature as implemented.

## Experiment placement

Always place Slurm scripts under `experiments/`, never under `src/`.
Profiling runners and benchmark configurations also belong under
`experiments/`; `src/` is for reusable library code. Preserve completed
result folders and their frozen source snapshots when reorganizing code.
