![](https://img.shields.io/badge/python-≥3.11-blue)

# Verifier-guided formula search

Code for the verifier-guided discovery of causal identifying formulas. The
project plan lives in the `formula-search-docs` repository; formulas are
written in the grammar of, and checked with,
[`hiprof`](https://github.com/francescofreni/hiprof-eq).

## ⚙️ Installation

`formula-search` needs `hiprof` as developed in `hiprof-eq`. The `hiprof`
release on PyPI is an earlier version with a different falsifier interface,
so install `hiprof-eq` first, from a clone next to this repository:

```bash
pip install -e ../hiprof-eq
pip install -e ".[dev]"
```

or directly from GitHub:

```bash
pip install "hiprof @ git+https://github.com/francescofreni/hiprof-eq.git"
pip install -e ".[dev]"
```

## 🚀 Baselines

`identify` runs the ID algorithm of Shpitser and Pearl (2006) and returns an
identifying formula in hiprof's grammar, or `None` if the target is not
identifiable:

```python
from hiprof import HPFalsifier
from formula_search.baselines import identify

graph = "T -> M; M -> Y; T <-> Y"
formula = identify(graph, treatments="T", outcomes="Y")
# sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }

HPFalsifier(graph).check("p(Y | do(T))", formula)
# True
# False-acceptance bound: 5.421e-18

identify("T -> Y; T <-> Y", treatments="T", outcomes="Y")
# None
```

ID can return expressions outside the grammar (Freni et al., 2026,
Example 7). `identify` rebuilds them as in the proof of their Proposition 1,
so that every formula it returns is admissible. With `conditions`, it runs
the IDC algorithm of Shpitser and Pearl (2008) instead:

```python
identify(graph, treatments="T", outcomes="Y", conditions="M")
# sum_{T'} { p(T') p(Y | M, T') }
```

`identify_equivalent` implements the method of Yvernes et al. (2026): it
applies IDC to every query that do-calculus proves equal to the target, and
returns the resulting formulas for the target, keyed by the query:

```python
from formula_search.baselines import identify_equivalent

identify_equivalent("W -> Z; Z -> X; X -> Y; W <-> Y", ["W", "Z"], "Y")
# {'p(Y | do(W, Z))': "sum_{X} { p(X | W, Z) sum_{W'} { p(W') p(Y | W', X, Z) } }",
#  'p(Y | do(Z))': "sum_{X} { p(X | Z) sum_{W'} { p(W') p(Y | W', X, Z) } }",
#  'p(Y | do(W), Z)': "sum_{X} { p(X | W, Z) sum_{W'} { p(W') p(Y | W', X, Z) } }"}
```

`template_formulas` fills the slots of the adjustment, front-door,
front-door with adjustment, and napkin formulas with disjoint sets of the
other variables, in every way, and returns each formula with the template
and slots that give it:

```python
from formula_search.baselines import template_formulas

template_formulas("T -> M; M -> Y; T <-> Y", "T", "Y")
# {'p(Y | T)': 'adjustment: S = {}',
#  'sum_{M} { p(M) p(Y | M, T) }': 'adjustment: S = {M}',
#  'p(Y)': 'front-door: S = {}',
#  "sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }": 'front-door: S = {M}'}
```

## ⚖️ Evaluator

An `Archive` keeps the distinct valid formulas for a task, one per class.
It checks with HiProF that each formula submitted identifies the target, and
whether it is a function of the observed law already archived, by checking
it against each class on the complete DAG, whose model imposes no
constraint. A new class records the single edges whose addition to the
graph breaks it:

```python
from formula_search.evaluator import Archive, Task

task = Task("Z1 -> T; Z1 -> Y; T -> Y; Z2 -> Y", treatment="T", outcome="Y")
archive = Archive(task)
for formula, source in template_formulas(*task).items():
    archive.submit(formula, source)

[(known.formula, sorted(known.broken_by)) for known in archive.classes][:2]
# [('sum_{Z1} { p(Z1) p(Y | T, Z1) }', ['T <-> Y', 'T <-> Z2', 'Z2 -> T']),
#  ('sum_{Z1, Z2} { p(Z1, Z2) p(Y | T, Z1, Z2) }', ['T -> Z2', 'T <-> Y'])]

archive.submit("p(Y | T)").outcome
# 'falsified'
```

`suite.CURATED` holds the curated tasks of the plan, and
`suite.random_tasks` draws random ones from a seed.

## 🌳 Derivation search

`derive` searches derivations of the target with Rules 2 and 3 of
do-calculus and the rules of probability, in increasing total rule cost,
and keeps one derivation per class. Transports between equivalent queries
are explicit moves and can compose. Each derivation can
be checked again step by step, and makes its formula derivation-certified in
an archive:

```python
from formula_search.derivation import derive

napkin = Task("W -> Z; Z -> X; X -> Y; W <-> X; W <-> Y", "X", "Y")
archive = Archive(napkin)
for derivation in derive(*napkin, bound=3, limit=3):
    archive.submit(derivation.formula, derivation)
    print(derivation.formula)
# sum_{Z} { p(Z) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }
# sum_{Z} { p(Z | X) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }
# sum_{W} { p(W | X) sum_{Z} { p(Z | W) icd_{X | Z} { sum_{W'} { p(W') p(X, Y | W', Z) } } } }
```

Probability rules cost one by default; base terms cost zero. Transport
cost is configured with `costs={"transport": 0}` (the default) or a positive
integer. Each transport wrapper costs once, regardless of the number of
do-calculus edges in its path. Probability-rule overrides remain positive
integers. `bound` limits the total cost; `derivation.cost(costs)` computes it.

Zero-cost composition can produce infinitely many distinct formula classes
at a finite bound. Use `limit` for a prefix, or positive transport cost for
finite bounded enumeration. Both engines default to `max_expansions=10000`;
hitting it raises `SearchLimitExceeded` with `.derivations` and `.expanded`,
never silently returning an exhaustive result. `max_expansions=None`
disables the guard and may not terminate with free transport. A class-limit
return is also only a prefix, not evidence that no other classes exist.

```python
# Finite bounded search: the four napkin classes reachable through cost 7.
derivations = derive(*napkin, bound=7, costs={"transport": 1})
```

[astar.py](src/formula_search/astar.py) configures heuristic-guided inference
using the same agenda and subproof-class deduplication as `derivation.py`:

```python
from formula_search.astar import derive as derive_astar

derivations = derive_astar(*napkin, bound=3, limit=3)
```

A query heuristic lower-bounds its cheapest proof cost (inside cost). The
agenda uses these estimates for sibling premises to lower-bound the context
needed to reach the target (outside cost). Admissible estimates preserve
minimum-cost discovery even when inconsistent. Built-in bounds relax
transport costs to zero; this remains admissible when transport costs more.
Only actual costs prune the bound. Small graphs and bounds are recommended.

Heuristics and tie-breakers can be mixed independently:

```python
from formula_search.heuristics import (
    conditioning_size,
    free_proof_bound,
    lookahead_bound,
    max_of,
    summation_count,
    term_count,
)

costs = {"marginalization": 2}
derivations = derive_astar(
    *napkin,
    bound=4,
    limit=3,
    costs=costs,
    heuristic=max_of(
        free_proof_bound(napkin.graph, costs=costs),
        lookahead_bound(napkin.graph, depth=2, costs=costs),
    ),
    tiebreaker=[
        term_count(),
        summation_count(weight=2),
        conditioning_size(),
    ],
)
```

`max_of` takes the largest bound, preserving admissibility when its inputs
are admissible. Use the same graph and rule costs in the factories and the
search. Remove bounds to exclude them; `max_of()` is zero. Tie-breaker
scores are summed, with smaller scores preferred only at equal A* priority.
Remove scorers, set their weights to zero, or use `tiebreaker=[]` to disable
all tie-breakers. `expression_size()` counts all expression nodes, and
`conditioning_count()` counts ICD operations.

A custom tie-breaker is a callable receiving a read-only `SearchState` with
the local inference's `query`, committed `cost`, `pending` queries, an
`expression`, and an optional
completed `derivation`. Unfinished expressions use kernel placeholders and
provisional transport weights; they estimate size and are not formulas.
Completed candidates use their actual simplified expressions. Tie-breaking
guides discovery; it does not guarantee globally smallest formulas.

## 🗂️ Layout

```
src/formula_search/
├── graph.py                     acyclic directed mixed graphs
├── docalculus.py                rules 2 and 3 of do-calculus
├── expression.py                kernel expressions in hiprof's grammar
├── evaluator.py                 archives of distinct valid formulas
├── suite.py                     curated and random tasks
├── derivation.py                derivation search
├── astar.py                     heuristic-guided AND/OR inference
├── heuristics.py                cost bounds and expression tie-breakers
└── baselines/
    ├── id.py                    the ID and IDC algorithms
    ├── equivalent_queries.py    ID applied to equivalent queries
    └── templates.py             formulas from templates
tests/
```

## 🛠️ Development

```bash
pytest
black --check src tests
ruff check src tests
mypy src
```
