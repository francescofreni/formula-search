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
so that every formula it returns is admissible.

## 🗂️ Layout

```
src/formula_search/
├── graph.py        acyclic directed mixed graphs
├── expression.py   kernel expressions in hiprof's grammar
└── baselines/
    └── id.py       the ID algorithm
tests/
```

## 🛠️ Development

```bash
pytest
black --check src tests
ruff check src tests
mypy src
```
