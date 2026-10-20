# Implementation progress

Snapshot of 2026-10-20, including uncommitted work. The plan this code
implements is in the sibling repository `formula-search-docs`: the stages in
its [README](../formula-search-docs/README.md#plan), the concrete steps in
[IMPLEMENTATION.md](../formula-search-docs/IMPLEMENTATION.md), and the
project state in [STATUS.md](../formula-search-docs/STATUS.md). Usage
examples for each public function are in the [README](README.md); this file
says what exists, how the derivation search works, and what comes next.

## At a glance

All work so far belongs to stage A of the plan, ground truth and non-learned
methods.

| Plan step | Content | Code | Status |
|---|---|---|---|
| Foundations | Graphs, do-calculus, kernel expressions | [graph.py](src/formula_search/graph.py), [docalculus.py](src/formula_search/docalculus.py), [expression.py](src/formula_search/expression.py) | Done |
| Existing methods | ID, IDC, ID on equivalent queries (Yvernes et al.) | [baselines/id.py](src/formula_search/baselines/id.py), [baselines/equivalent_queries.py](src/formula_search/baselines/equivalent_queries.py) | Done |
| [A1](../formula-search-docs/IMPLEMENTATION.md#a1-evaluator) | Evaluator | [evaluator.py](src/formula_search/evaluator.py), [fingerprints.py](src/formula_search/fingerprints.py) | Done, without an `unresolved` outcome ([why](#evaluator-a1)) |
| [A2](../formula-search-docs/IMPLEMENTATION.md#a2-graph-suite-and-template-enumeration) | Graph suite, template enumeration | [suite.py](src/formula_search/suite.py), [baselines/templates.py](src/formula_search/baselines/templates.py) | Done on the curated graphs; labels, families, and size bounds missing |
| [A3](../formula-search-docs/IMPLEMENTATION.md#a3-baseline-archive-b) | Baseline archive B | none | Not started; do-search set aside for now |
| [A4](../formula-search-docs/IMPLEMENTATION.md#a4-id-on-weakened-graphs) | ID on weakened graphs | none | Not started; set aside for now |
| [A5](../formula-search-docs/IMPLEMENTATION.md#a5-bottom-up-enumeration) | Bottom-up enumeration | none | Not started |
| [A6](../formula-search-docs/IMPLEMENTATION.md#a6-derivation-search) | Derivation search | [derivation.py](src/formula_search/derivation.py) | Engine done; frontier and diagnosis missing; class explosion open |

No experiment has been run and no archive has been saved yet.

## What each module does

### Foundations

- **[graph.py](src/formula_search/graph.py).** `ADMG`, an acyclic directed
  mixed graph, parsed from and written in hiprof's syntax
  (`"T -> M; M -> Y; T <-> Y"`). It provides ancestors, districts,
  mutilation, m-separation (Richardson, 2003), and the complete DAG over the
  nodes that the evaluator uses as its saturated graph. `closure` and
  `implies_constraints` decide whether a graph implies any equality
  constraint (Bhattacharya et al., 2022, Algorithm 1 and Theorem 1). A graph
  that implies none has a single class of identifying formulas, which makes
  it a negative control for discovery.
- **[docalculus.py](src/formula_search/docalculus.py).** `Query`, the query
  `p(outcomes | do(treatments), conditions)`; `rule2` and `rule3` as
  graphical tests; `neighbours`, the queries one rule on one variable away;
  and `equivalent_queries`, a breadth-first search returning the connected
  component of the derivation graph of Yvernes et al. that contains a query,
  with a shortest path to each member. Rule 1 follows from Rules 2 and 3.
- **[expression.py](src/formula_search/expression.py).** Kernel expressions
  in hiprof's grammar: `Term` (`p(A | B)`), `Product`, `Sum`, and `ICD`. Each
  has typed `outputs` and `inputs`. The constructors `product`, `marginal`,
  `condition`, and `average` build admissible expressions and simplify them
  with identities valid for every positive law. `render` writes hiprof
  syntax and renames summed copies of variables already in use to `T'`.

### Existing methods

- **[baselines/id.py](src/formula_search/baselines/id.py).** `identify` runs
  ID (Shpitser and Pearl, 2006), or IDC (2008) when it gets `conditions`.
  When ID's product is not admissible in the grammar, it is rebuilt as in
  the proof of Proposition 1 of Freni et al. (2026). The tests match the
  fixing criterion for identifiability on 300 random graphs, and unlike y0
  0.2.11 it returns a valid formula on the napkin graph.
- **[baselines/equivalent_queries.py](src/formula_search/baselines/equivalent_queries.py).**
  `identify_equivalent` applies IDC to every query in the target's
  component. Inputs that the target lacks are averaged out with a weight
  `p(S | C)`, where `C` holds the formula's other inputs.
- **[baselines/templates.py](src/formula_search/baselines/templates.py).**
  `template_formulas` fills the adjustment, front-door, front-door with
  adjustment, and napkin templates with disjoint sets of the other
  variables, in every way. That is (k + 1)^n fillings for k slots and n
  other variables. Each formula is returned with its template and slots.

### Evaluator (A1)

[evaluator.py](src/formula_search/evaluator.py) and
[fingerprints.py](src/formula_search/fingerprints.py). A `Task` is a graph
with a treatment and an outcome. An `Archive` holds the classes of valid
formulas found for one task. `Archive.submit(formula, source)` returns a
`Verdict` and runs its checks cheapest first:

1. **Malformed:** the formula does not parse, uses `do`, or is not a kernel
   of the outcome given at most the treatment. The reason is recorded.
2. **Falsified:** HiProF rejects it as equal to the target in the graph.
3. **Duplicate:** HiProF accepts it as equal, on the complete DAG, to a
   formula already archived. The verdict names that class. Only the classes
   with the same fingerprint are checked: the formula's value, in exact
   rationals, on one linear Gaussian model of the complete DAG, which
   `FixedModel` draws the same in every run. Formulas of one class share a
   fingerprint, and formulas with different fingerprints are different
   functions, so a duplicate takes one such check and a new class none,
   however many classes there are. A fingerprint never decides that two
   formulas are equal, and an identical formula needs no check.
4. **New:** it opens a class, a `FormulaClass`. The class records the
   formula's source, HiProF's evidence, the single edges whose addition to
   the graph breaks the formula (`broken_by`), which is the assumption
   profile, and the formula's `features`: the number of base terms, the
   variables summed out, the largest conditioning set, and whether it has a
   quotient or an `icd`.

`archive.checks` counts HiProF calls, for a union bound on false
acceptances. A formula submitted with a `Derivation` as its source is
replayed first: a derivation of another formula, or one with an invalid
step, raises `ValueError`. A formula that passes is `verdict.certified`,
that is, derivation-certified, and so is a class whose source is a
derivation. Any other valid formula is verification-supported.

`archive.save(path)` writes the task, the checks, the classes, and the log
of every verdict as JSON, with each derivation as its tree of rules and
each query in hiprof's syntax. This covers the
[archive records](../formula-search-docs/IMPLEMENTATION.md#archive-records)
of the plan, except that a class is identified by its index, its profile
lists only the breaking edges, since the others follow from the graph, and
the HiProF commit and the run belong to the run log of A3.

There is no `unresolved` outcome, which the plan reserves for constructs
HiProF does not support, timeouts, and other stated limitations. HiProF
decides every formula its grammar accepts, and the plan counts parse and
validation errors as malformed, so only a timeout would leave a formula
unresolved. A check takes about 0.3 ms, and a time limit within one process
needs Unix signals, which other libraries can intercept. A run can instead
bound the time of each task in a worker process, and log what it cuts off
as unresolved (A3).

The fingerprints and features use modules that hiprof does not export
(`hiprof.verification.gaussian`, `hiprof.formula.formula`, and
`hiprof.formula.parser`), so a change to them in hiprof-eq can break both.
The tests would show it.

### Graph suite (A2)

[suite.py](src/formula_search/suite.py). `CURATED` holds 13 tasks: front-door,
napkin, two adjustment sets, confounder and mediator, collider and
mediator, adjustment-rich, ADMGs (ii) to (iv) of `hiprof-eq`, the split
product of Freni et al. (Example 7), Figure 15 of Yvernes et al., Sachs, and
the bow arc as a non-identifiable control. `random_tasks(count, size,
density, seed)` draws reproducible random graphs on which ID identifies
`p(Y | do(T))` and `T` is an ancestor of `Y`.

The tests reproduce A2's checks: one class on the front-door graph, 48
front-door classes on ADMG (iv), 14 adjustment classes on the
adjustment-rich graph, and identical archives on repeated runs.

## Derivation search

[derivation.py](src/formula_search/derivation.py) implements step A6. It
extends the derivation graph of Yvernes et al., whose edges are do-calculus
steps, with rules of probability. With those rules it can reach formulas
that ID cannot, on any query in the component. Every formula it returns
comes with a derivation that can be replayed step by step.

### The search space

The nodes are queries `p(a | do(x), b)` over the graph's variables. There
are finitely many of them, even though the expressions derived are
infinitely many (Yvernes et al., Section 6). A query is reached in three
ways:

- **Base term.** A query without interventions, `p(a | b)`, is its own
  observational term, at no cost.
- **Rules of probability,** listed by
  [`rules(graph, query)`](src/formula_search/derivation.py#L157). Each
  writes the query in terms of other queries, called its premises, and
  costs 1:

  | Rule | Identity | Premises |
  |---|---|---|
  | marginalization | `p(a \| do(x), b) = sum_{z} { p(a, z \| do(x), b) }` | one, for each variable `z` not in the query |
  | conditioning | `p(a \| do(x), b, c) = icd_{c \| b, x} { p(a, c \| do(x), b) }` | one, for each condition `c` |
  | chain rule | `p(a, c \| do(x), b) = p(a \| do(x), b) p(c \| do(x), a, b)` | two, for each split of the outcomes |

  Marginalization and conditioning act on one variable at a time, since
  they compose.
- **Do-calculus,** at no cost. Every query in the component of the current
  query shares its derivations. If a derivation of a member `q'` uses
  inputs that the query `q` lacks, for example `Z` in
  `p(Y | do(X)) = p(Y | do(X, Z))`, those inputs are averaged out with a
  weight `p(extra | C)`, one for each subset `C` of the inputs of `q`
  ([`_transport`](src/formula_search/derivation.py#L251)). Every weight
  gives a valid formula, because the expression does not depend on those
  inputs in the model. Different weights can still give different classes,
  as on the napkin graph.

As in HyperTree Proof Search, this is an AND/OR graph: one rule suffices to
derive a query, and the chain rule needs both of its premises derived.

### The algorithm

The **cost** of a derivation is its number of rules of probability.
`derive` asks the search for the derivations of the target at costs
0, 1, …, `bound`. The search is
[`_Search`](src/formula_search/derivation.py#L179), which runs two mutually
recursive, memoized functions:

```
layer(q, c):   derivations of q with cost c, one per class not reached by cost < c
    compute layer(q, c') for every c' < c          # cheaper classes register first
    for each member q' of the component of q, with its do-calculus path from q:
        for each d in direct(q', c):
            for each d* in transport(d, q, path):   # one per weight, if needed
                keep d* if its formula is a new class of q

direct(q, c):  derivations whose last rule applies to q itself
    c = 0:  [base(q)] if q has no do(), else []
    c > 0:  for each rule of probability q <- (p1, ..., pk):
                for each split c1 + ... + ck = c - 1:
                    for each choice d1 in layer(p1, c1), ..., dk in layer(pk, ck):
                        Derivation(q, rule, (d1, ..., dk))
```

`direct(q, c)` only asks for layers of cost below `c`, so the recursion
ends, and cycles among queries cost nothing extra. Components are computed
once per query, and every `(query, cost)` pair is computed once.

**Classes.** [`_new_class`](src/formula_search/derivation.py#L236) keeps a
derivation only if HiProF, run on the complete DAG, rejects its formula as
equal to every formula already kept for that query with the same
fingerprint. This is the evaluator's distinctness test, applied to every
subquery. No class of the target is lost, since substituting formulas of
one class into the same rule gives formulas of one class.

HiProF is used only to deduplicate here, never to accept. Validity comes
from the rules. So a wrong "same class" verdict, with probability at most
about 1e-14 per check, could drop a class, but it could never let an
invalid formula through.

### Arguments

```python
derive(graph, treatments, outcomes, bound) -> list[Derivation]
```

| Argument | Type | Meaning |
|---|---|---|
| `graph` | `str` | ADMG in hiprof's syntax, such as `"T -> M; M -> Y; T <-> Y"` |
| `treatments` | `str` or iterable of `str` | The intervened variables `x` of `p(y \| do(x))` |
| `outcomes` | `str` or iterable of `str` | The outcomes `y`, not empty |
| `bound` | `int` | The largest number of rules of probability in a derivation. Do-calculus steps and base terms are free |

A `Task` unpacks into the first three arguments: `derive(*task, bound=4)`.
The function returns one derivation per class of the target found, in order
of cost. It raises `ValueError` for an invalid graph or invalid variables,
as `identify` does.

An empty list only means that no derivation was found within the bound. It
does not show that the target is not identifiable. Unlike `identify`,
`derive` takes no `conditions`. The smallest bounds that find a formula are
2 for adjustment, 3 for the napkin formula, and 4 for the front-door formula.

### The output: `Derivation`

A [`Derivation`](src/formula_search/derivation.py#L52) is a frozen tree of
rules.

| Field | Meaning |
|---|---|
| `query` | The `Query` derived at this node |
| `rule` | `"base"`, `"marginalization"`, `"conditioning"`, `"chain rule"`, or `"do-calculus"` |
| `premises` | Sub-derivations of the queries the rule uses: none for `base`, two for the chain rule, one otherwise |
| `path` | For `do-calculus`: the queries from `query` to the premise's query, each one rule from the last |
| `weight` | For `do-calculus`: the conditioning set `C` of the weight `p(extra \| C)`, when the premise has inputs that `query` lacks |
| `expression` | The derived `Expression` tree (`Term`, `Product`, `Sum`, `ICD`), computed on demand |
| `formula` | `expression` in hiprof's syntax |
| `verify(graph)` | Replays the tree. It checks that every probability step is among `rules(graph, query)`, that every do-calculus path is a chain of `neighbours` with the side conditions tested again, that base terms have no interventions, and that every node's expression has its query's type |

### Inspecting the output

The formulas themselves:

```python
from formula_search.derivation import derive
from formula_search.suite import CURATED

task = CURATED["napkin"]
derivations = derive(*task, bound=3)
[d.formula for d in derivations]
# ['sum_{Z} { p(Z) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }',
#  'sum_{Z} { p(Z | X) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }']
```

There is no built-in printer for the tree; this helper prints one rule per
line:

```python
from formula_search.derivation import Derivation
from formula_search.expression import render_query


def show(derivation: Derivation, depth: int = 0) -> None:
    """Print a derivation as a tree, one rule per line."""
    line = f"{'  ' * depth}{derivation.query}  [{derivation.rule}]"
    if derivation.rule == "do-calculus":
        line += "  " + " = ".join(map(str, derivation.path[1:]))
        (premise,) = derivation.premises
        if extra := premise.expression.inputs - derivation.query.inputs:
            weight = render_query(extra, conditions=derivation.weight)
            line += f", weight {weight}"
    print(line)
    for premise in derivation.premises:
        show(premise, depth + 1)


show(derivations[1])
# p(Y | do(X))  [do-calculus]  p(Y | do(X, Z)) = p(Y | do(Z), X), weight p(Z | X)
#   p(Y | do(Z), X)  [marginalization]
#     p(W, Y | do(Z), X)  [conditioning]
#       p(W, X, Y | do(Z))  [chain rule]
#         p(W | do(Z))  [do-calculus]  p(W)
#           p(W)  [base]
#         p(X, Y | do(Z), W)  [do-calculus]  p(X, Y | W, Z)
#           p(X, Y | W, Z)  [base]
```

Read the tree from the top. The target equals `p(Y | do(Z), X)` by
do-calculus: Rule 3 adds `do(Z)`, then Rule 2 turns `do(X)` into observing
`X`. That query depends on `Z`, which the target lacks, so `Z` is averaged
out with the weight `p(Z | X)`. The weight `p(Z)` gives the first formula.
The two are different functions because of the napkin's Verma constraint.
The three rules of probability give the cost, 3.

To replay a derivation and record it in an archive:

```python
from formula_search.evaluator import Archive
from formula_search.graph import ADMG

derivations[1].verify(ADMG.parse(task.graph))   # True

archive = Archive(task)
for derivation in derivations:
    verdict = archive.submit(derivation.formula, derivation)
    print(verdict.outcome, verdict.class_index, verdict.certified)
# new 0 True
# new 1 True
sorted(archive.classes[1].broken_by)
# ['W -> X', 'W -> Y', 'W <-> Z', 'X <-> Y', 'X <-> Z', 'Y <-> Z', 'Z -> Y']
```

To debug the search itself, for example to see which subqueries have many
classes, use the private `_Search` directly. Its interface may change:

```python
from formula_search.derivation import _Search
from formula_search.docalculus import Query

search = _Search(ADMG.parse(task.graph))
search.layer(Query(frozenset({"Y"}), frozenset({"X"})), 3)
len(search.components)                                   # queries visited
sorted((sum(map(len, c.values())), str(q)) for q, c in search.classes.items())[-5:]   # largest
```

### Measured behaviour

New classes of the target at each cost, measured on 2026-10-20 on a laptop
with nine runs in parallel. The times are cumulative and indicative; a dash
marks a cost not run.

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| front-door | 0 | 0 | 0 | 0 | 1 | 0 | 0, all under 0.1 s |
| napkin | 0 | 0 | 0 | 2 | 0 | 4 | 0, 0.2 s in all |
| two adjustment sets | 0 | 0 | 2 | 2 | 173, 1.8 s | 156, 5.1 s | – |
| split product | 0 | 0 | 0 | 0 | 1, 0.6 s | 0, 4.3 s | 93, 505 s |
| front-door (ii) | 0 | 0 | 0 | 0, 0.2 s | 8,135, 148 s | – | – |
| front-door (iii) | 0 | 0 | 0 | 0, 2.6 s | not in 15 min | – | – |
| front-door (iv) | 0 | 0 | 0, 0.9 s | not in 15 min | – | – | – |
| adjustment-rich | 0 | 0 | 0 | 73, 35 s | not in 15 min | – | – |
| Figure 15 of Yvernes et al. | 0 | 0 | 0 | 0, 21 s | not in 15 min | – | – |

Fingerprints change no class at any cost compared with the code of
2026-10-09, which checked every candidate against every class of its query.
They cut the HiProF checks, and the time, on the same machine and load:

| Task, up to cost | Checks before | Checks after | Time before | Time after |
|---|---|---|---|---|
| two adjustment sets, 4 | 111,805 | 955 | 91.6 s | 1.8 s |
| split product, 5 | 169,576 | 1,558 | 175 s | 4.3 s |

- **Recovery of B.** At cost 4 on two adjustment sets, the search finds
  ID's class, `sum_{Z1, Z2} { p(Z1) p(Z2) p(Y | T, Z1, Z2) }`. On the split
  product, its one class at cost 4 is the front-door formula over `A`. That
  formula is valid and differs from ID's longer split formula, but ID's own
  class is not reached by cost 5. The tests check recovery of B only on
  front-door, napkin, confounder and mediator, and collider and mediator.
- **Frontier.** The front-door formula needs cost 4. On front-door (ii),
  cost 4 finishes in 148 s with 8,135 classes. On front-door (iii) and (iv),
  adjustment-rich, and Figure 15 of Yvernes et al., which have 6 to 9
  variables, the cost after the last in the table does not finish within
  15 minutes. Adjustment-rich has 73 classes at cost 3, such as adjustment
  for `{A, C}`, all valid and replayed. The snapshot of 2026-10-09 reported
  none there, but with the code of that date cost 3 does not finish within
  15 minutes.
- **Class explosion.** Wherever the graph implies independences, subqueries
  gain many classes that differ only by those independences. For example,
  `p(Z1)` is also derived as `sum_{Z2} { p(Z2 | T) p(Z1 | Z2) }`, which is
  valid in G but a different function on the complete DAG. Combining them
  through the chain rule multiplies their numbers. Most of the 173 classes
  at cost 4 on two adjustment sets are such variants, and front-door (ii),
  where B and the templates give 14 classes together, has 8,135 at cost 4.
  An earlier experiment (2026-10-07) found that restricting marginalization,
  chain-rule order, or weights cuts the 173 classes only to about 111.
  Fingerprints removed the cost of deduplication, but the number of
  candidates still grows with the classes they combine.

### Gaps relative to step A6

- No optional IDC move, which would solve a query in one step (A6.3).
- Weights are only `p(extra | C)` with `C` a subset of the query's inputs.
- No conditional targets.
- No frontier record, and no diagnosis of the template classes it misses as
  *search order* or *missing moves* (A6.6). That diagnosis decides what B3
  evolves.
- No comparison with do-search (A6.7).
- Rule steps are tested against HiProF on 30 random graphs with 5 nodes,
  with 3 random queries each. Certificates are replayed only by `verify`,
  in the same code. Independent replay in the UCPH rule checker is
  [step D2](../formula-search-docs/IMPLEMENTATION.md#d2-independent-replay).

## Next steps

### Decisions that block the rest

1. **What counts as a class**
   ([Q-006](../formula-search-docs/QUESTIONS.md#q-006-substantively-different-formulas)).
   Under the complete-DAG test, formulas that differ only by independences
   of G are distinct classes. This makes the derivation search explode, and
   it inflates "classes beyond B". The options are:
   - keep the test and report results at several levels, such as grouping
     classes by assumption profile;
   - adopt a coarser test for subqueries or for the target;
   - restrict the moves, which so far trims little.

   The choice fixes A6's frontier and what stage B has to beat.
2. **What B contains**
   ([Q-016](../formula-search-docs/QUESTIONS.md#q-016-set-valued-corpus-and-benchmark-validity)):
   which weights, which fixing orders, and whether do-search is included.
3. **The false-acceptance policy**
   ([Q-015](../formula-search-docs/QUESTIONS.md#q-015-global-false-acceptance-control)),
   before large runs.

I recommend using the A3 numbers below to decide 1.

### Implementation, in order

1. **Commit** the evaluator's features, fingerprints, and records, and
   update `formula-search-docs/STATUS.md`, which still lists the evaluator as
   next.
2. **A3, the first result.** A runner that passes `identify`,
   `identify_equivalent`, and `template_formulas` through the evaluator on
   every curated task and a set of random tasks. It saves each archive with
   `Archive.save`, writes a run log of the HiProF commit, seed, and check
   count, and reports the template classes outside B. Running each task in
   a worker process with a time budget would also give the `unresolved`
   outcome.
3. **Finish A2:** label each task by `implies_constraints`, group random
   graphs into isomorphism families for the development and test split,
   bound slot sizes on larger graphs such as Sachs, and add the examples of
   Raichev et al.
4. **Finish A6:** record the frontier per task and bound, check recovery of
   B on every curated graph, diagnose the missed template classes as search
   order or missing moves, and compare with do-search. All of this depends
   on decision 1.
5. **A5 and A4,** bottom-up enumeration and ID on weakened graphs, when
   they are taken up again.
6. **Stage B.** B1 (plain LLM sampling) needs only the evaluator, whose
   `features` also check its feature requests. B2 (guided derivation
   search) reuses `rules` and `_Search` from this module, ranking the edges
   of each query instead of expanding them breadth-first.
