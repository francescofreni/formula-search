# Distance between identifying formulas

Edit `CONFIG` in `configs.py`, then run from the repository root:

```bash
.venv/bin/python -m distance_metric
```

With your virtual environment activated, `python distance_metric/` from
`src/`, or `python .` from `src/distance_metric/`, also works.

The default, `ADJUSTMENT_03_AUTOMATIC`, compares adjustment by A versus
adjustment by B on `A -> B; B -> X; A -> Y; X -> Y`. It automatically
generates all missing forward and bidirected edges for each sampled valid
variable order. This graph happens to have only one valid order, but the
sampler still uses the same rejection procedure on every chain.

Two explicit-family controls remain available:

- `ADJUSTMENT_01_A_VS_B`: only `A -> X` and `B -> Y`; exact distance **1/2**.
- `ADJUSTMENT_02_SHARED_CONFOUNDING`: those two edges plus `X <-> Y`;
  exact distance **1/3**. Two of the six edge orders break both formulas
  immediately (distance 0); the other four give distance 1/2.

These known expectations describe their restricted edge families, not the
new automatic experiment. Select a preset at the bottom of `configs.py`.

## Config and layout

`formula_search.verification.VerificationSettings` shares graph, seed,
and per-check verifier tolerance. Single-equation HiProF profiling uses
`VerificationConfig`, adding `formula` and `target`; its API is unchanged.
`DistanceConfig` instead adds a target and a `formulas` collection, along
with the edge-generation and sampling settings.

**Config migration:** replace `formula=a, other_formula=b` with
`formulas=(a, b)`. Supply a tuple or list of **at least two** nonempty formula
strings; fewer entries raise `ValueError`. Lists are copied to immutable
tuples. Entries are not deduplicated: repeated formulas remain repeated
submissions and their pairwise distance is zero when verdicts are exact.

```python
from distance_metric import DistanceConfig, estimate_distance

config = DistanceConfig(
    graph="A -> B; B -> X; A -> Y; X -> Y",
    target="p(Y | do(X))",
    formulas=(
        "sum_{A} { p(A) p(Y | A, X) }",
        "sum_{B} { p(B) p(Y | B, X) }",
        "sum_{A, B} { p(A, B) p(Y | A, B, X) }",
    ),
    epsilon=0.05,
    delta=0.01,
    seed=0,
    target_bound=1e-14,
)
result = estimate_distance(config)
print(result.distance, result.interval, result.elapsed_seconds)
```

`ADJUSTMENT_04_THREE_FORMULAS` is the ready-to-run three-formula preset.
The result and logs include the number of formulas and distinct index pairs.

`main.py` only runs and reports. `edges.py` resolves and validates the
edge family; `estimator.py` handles binary search and sampling. No files, archived results, or network requests
are needed at runtime.

## Automatic optional edges

Omit both `optional_edges` and `topological_order`. For **each chain**:

1. Draw a uniform random permutation of **all observed variables**.
2. Reject it if any existing directed edge points backward in the permutation;
   draw a fresh full permutation and repeat until one is valid.
3. Generate every missing forward directed edge for the accepted order,
   plus every missing bidirected edge when `include_bidirected=True` (default).
4. Uniformly shuffle those optional edges and add them one at a time.

This rejection procedure samples valid topological orders uniformly. It
has no retry cap, fallback algorithm, or reuse of the preceding chain's
order. If there are `L` valid orders among `n!` permutations, each accepted
order takes `n! / L` attempts on average. This can be expensive on larger,
tightly ordered graphs; that cost is included in timing and attempt counts.

A bidirected edge such as `X <-> Y` represents an unobserved common cause.
It adds no observed variable and does not constrain topological order.
Set `include_bidirected=False` to exclude new bidirected additions; existing
ones remain in the graph. The directed part of every chain stays acyclic.

For controlled experiments only, an explicit `topological_order` fixes the
order, while an explicit `optional_edges` tuple fixes the edge family and
overrides generation and its bidirected flag. `()` means no additions.
Supplied orders must respect all supplied directed edges. These overrides
change the distance's sampling distribution and are identified in the logs.
`resolve_edges(config)` resolves a fixed experiment; automatic experiments
must first supply an accepted order to that helper. It never silently picks
an alphabetical order for an automatic experiment.

Both runners report the sampling procedure. After each estimate they print
attempted/accepted/rejected permutation counts, each distinct accepted order,
its chain count, and its generated edges. The result also retains sampling
mode, total attempts, and per-order counts. Settings and seeds are printed.

## What is estimated

By default the target is expected chain Jaccard distance over **uniform
valid topological orders, then uniform edge permutations conditional on
that order**. Different orders can generate different edge orientations.
For example, if A and B have no directed path between them, some orders
allow `A -> B` and others allow `B -> A`.

All chains contain the same number `m` of optional edges: each order has
`n*(n-1)/2` forward pairs, minus existing directed edges, plus the same set
of missing bidirected edges when enabled. A chain contains positions
**0 through m**, including the original graph. Orders and permutations are
sampled with replacement; samples are IID. Explicit-family controls instead
sample only permutations of their fixed edge lists.

For each formula, binary search finds the first invalid position, `k`.
Use `m + 1` if it stays valid throughout. Because the starting graph is
valid, `k` is also the number of valid positions. The chain distance is

```text
d_ij = abs(k_i - k_j) / max(k_i, k_j)
```

This is Jaccard distance between two valid prefixes. For `q` formulas,
compute each boundary **once per chain**, then score that chain by

```text
chain_score = sum(d_ij for i < j) / (q * (q - 1) / 2)
```

The reported distance averages `chain_score` over all sampled chains.
Equivalently it estimates the average expected distance over all formula
pairs. For **two formulas**, there is one pair and this recovers the usual
metric. Reordering the formulas leaves the mathematical score unchanged.
Verification is linear in `q` per chain; pairwise arithmetic is quadratic.

The target is expected chain Jaccard diversity, not a ratio of pooled
counts or Jaccard distance over uniformly chosen graphs. It measures the
collection's diversity, not its number of discoveries or novelty relative
to an external archive. Fix the formula collection before evaluation;
feedback-driven selection requires fresh evaluation chains. The graph,
target, edge-generation settings, and sampling distribution must be fixed
when comparing scores.

## Error bounds

The chain count is derived, not chosen independently:

```text
N = ceil(log(2 / delta) / (2 * epsilon**2))
```

Defaults `epsilon=0.05`, `delta=0.01` give **1,060 chains**. With exact
validity answers, Hoeffding gives probability at most 0.01 that the estimate
is off by 0.05 or more. The reported interval is clipped to [0, 1].
Each chain's group score lies in [0, 1], so the same bound and 1,060-chain
budget apply regardless of formula count. This is a guarantee on the
**aggregate score**, not a simultaneous guarantee on every pair.

HiProF answers are probabilistic. We separately bound any false acceptance
by `Q_max * target_bound`, clipped at 1, where

```text
Q_max = q * (1 + N * ceil(log2(m + 1)))
```

Here `q` is the formula count; the first `q` calls check all formulas on
the base graph. For `q=2` this is the previous budget. We use this fixed
worst-case budget, not the random realized call count, for the union bound.
The combined failure bound is at most `delta + Q_max * target_bound`,
clipped at 1. Thus the default combined bound is slightly above 1%, not
exactly 1%. All validity and distance guarantees apply within HiProF's
linear-Gaussian model family, not unrestricted causal models.

## Validation and timing

Optional edges must be distinct, absent from the base graph, and connect
existing observed variables. Both directed and bidirected additions are
supported. All additions together must remain acyclic in their directed
part, ensuring every order is valid. Cyclic orders are rejected by rejecting
the whole family, rather than resampling and changing the distribution.
All observational formulas must pass on the starting graph; otherwise the
estimator raises an error. This ensures the Jaccard denominator is nonzero.

There is no verdict cache beyond the base-graph checks. Every queried
position creates a fresh HiProF verifier. Total time includes validation,
sampling, graph construction, and verification, but excludes imports and
printing. The result reports both actual and worst-case verifier calls.

Topological-order, edge-order, and verifier draws use separate reproducible
seeded streams. The
call restores Python's global RNG state afterward. Use sequential execution:
HiProF draws from the global RNG, so concurrent threads could interfere.
