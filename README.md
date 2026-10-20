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
it on the complete DAG, whose model imposes no constraint, against the
classes with the same fingerprint: its value on a fixed model of that DAG.
A new class records the single edges whose addition to the graph breaks it,
how its formula was found, and the formula's features:

```python
from formula_search.evaluator import Archive, Task

task = Task("Z1 -> T; Z1 -> Y; T -> Y; Z2 -> Y", treatment="T", outcome="Y")
archive = Archive(task)
for formula, source in template_formulas(*task).items():
    archive.submit(formula, source)

[(known.formula, sorted(known.broken_by)) for known in archive.classes][:2]
# [('sum_{Z1} { p(Z1) p(Y | T, Z1) }', ['T <-> Y', 'T <-> Z2', 'Z2 -> T']),
#  ('sum_{Z1, Z2} { p(Z1, Z2) p(Y | T, Z1, Z2) }', ['T -> Z2', 'T <-> Y'])]

archive.classes[0].source
# 'adjustment: S = {Z1}'
archive.classes[0].features
# Features(kernels=2, summed=('Z1',), largest_conditioning_set=2, quotient=False, icd=False)

archive.submit("p(Y | T)").outcome
# 'falsified'
```

`archive.save(path)` writes the task, the number of HiProF checks, the
classes, and the verdict on every formula submitted, as JSON.

`suite.CURATED` holds the curated tasks of the plan, and
`suite.random_tasks` draws random ones from a seed.

## 🌳 Derivation search

`derive` searches derivations of the target with Rules 2 and 3 of
do-calculus and the rules of probability, breadth-first in the number of
rules of probability, and keeps one derivation per class. Each derivation can
be checked again step by step, and makes its formula derivation-certified in
an archive:

```python
from formula_search.derivation import derive

napkin = Task("W -> Z; Z -> X; X -> Y; W <-> X; W <-> Y", "X", "Y")
archive = Archive(napkin)
for derivation in derive(*napkin, bound=3):
    archive.submit(derivation.formula, derivation)
    print(derivation.formula)
# sum_{Z} { p(Z) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }
# sum_{Z} { p(Z | X) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }
```

## 🗂️ Layout

```
src/formula_search/
├── graph.py                     acyclic directed mixed graphs
├── docalculus.py                rules 2 and 3 of do-calculus
├── expression.py                kernel expressions in hiprof's grammar
├── fingerprints.py              values of formulas on a fixed model
├── evaluator.py                 archives of distinct valid formulas
├── suite.py                     curated and random tasks
├── derivation.py                derivation search
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
