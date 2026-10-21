# Implementation progress

Snapshot of 2026-10-09, including uncommitted work. The plan this code
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
| [A1](../formula-search-docs/IMPLEMENTATION.md#a1-evaluator) | Evaluator | [evaluator.py](src/formula_search/evaluator.py) | Core done; features, fingerprints, and stored records missing |
| [A2](../formula-search-docs/IMPLEMENTATION.md#a2-graph-suite-and-template-enumeration) | Graph suite, template enumeration | [suite.py](src/formula_search/suite.py), [baselines/templates.py](src/formula_search/baselines/templates.py) | Done on the curated graphs; labels, families, and size bounds missing |
| [A3](../formula-search-docs/IMPLEMENTATION.md#a3-baseline-archive-b) | Baseline archive B | none | Not started; do-search set aside for now |
| [A4](../formula-search-docs/IMPLEMENTATION.md#a4-id-on-weakened-graphs) | ID on weakened graphs | none | Not started; set aside for now |
| [A5](../formula-search-docs/IMPLEMENTATION.md#a5-bottom-up-enumeration) | Bottom-up enumeration | none | Not started |
| [A6](../formula-search-docs/IMPLEMENTATION.md#a6-derivation-search) | Derivation search | [derivation.py](src/formula_search/derivation.py) | Engine done; frontier and diagnosis missing; class explosion open |

No experiment has been run and no archive has been written to disk yet.

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

[evaluator.py](src/formula_search/evaluator.py). A `Task` is a graph with a
treatment and an outcome. An `Archive` holds the classes of valid formulas
found for one task. `Archive.submit(formula, source)` returns a `Verdict`
and runs its checks cheapest first:

1. **Malformed:** the formula does not parse, uses `do`, or is not a kernel
   of the outcome given at most the treatment. The reason is recorded.
2. **Falsified:** HiProF rejects it as equal to the target in the graph.
3. **Duplicate:** HiProF accepts it as equal, on the complete DAG, to a
   formula already archived. The verdict names that class.
4. **New:** it opens a class. The class records the single edges whose
   addition to the graph breaks the formula (`broken_by`), which is the
   assumption profile.

`archive.checks` counts HiProF calls, for a union bound on false
acceptances. A formula submitted with a `Derivation` as its source is
replayed first: a derivation of another formula, or one with an invalid
step, raises `ValueError`. A formula that passes is `verdict.certified`,
that is, derivation-certified. Any other valid formula is
verification-supported.

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
  [`rules(graph, query)`](src/formula_search/derivation.py#L155). Each
  writes the query in terms of other queries, called its premises, and
  defaults to cost 1:

  | Rule | Identity | Premises |
  |---|---|---|
  | marginalization | `p(a \| do(x), b) = sum_{z} { p(a, z \| do(x), b) }` | one, for each variable `z` not in the query |
  | conditioning | `p(a \| do(x), b, c) = icd_{c \| b, x} { p(a, c \| do(x), b) }` | one, for each condition `c` |
  | chain rule | `p(a, c \| do(x), b) = p(a \| do(x), b) p(c \| do(x), a, b)` | two, for each split of the outcomes |

  Marginalization and conditioning act on one variable at a time, since
  they compose.
- **Do-calculus transport,** at configurable cost (default zero). Every query in the component of the current
  query shares its derivations. If a derivation of a member `q'` uses
  inputs that the query `q` lacks, for example `Z` in
  `p(Y | do(X)) = p(Y | do(X, Z))`, those inputs are averaged out with a
  weight `p(extra | C)`, one for each subset `C` of the inputs of `q`
  ([`_transport`](src/formula_search/derivation.py#L243)). Every weight
  gives a valid formula, because the expression does not depend on those
  inputs in the model. Different weights can still give different classes,
  as on the napkin graph.

As in HyperTree Proof Search, this is an AND/OR graph: one rule suffices to
derive a query, and the chain rule needs both of its premises derived.

### The algorithm

The probability-rule costs default to one; `transport` defaults to zero.
Both are configurable, with positive integer probability costs and
nonnegative integer transport cost. A transport wrapper may prove a
multi-edge do-calculus path: it is charged once per wrapper. Base terms
are free. `Derivation.cost(costs)` sums these costs over the proof tree,
including repeated premises even when their computation was shared.

The shared `_Search` agenda in `derivation.py` works in two stages:

1. Backward reachability catalogs queries and inferences within a relaxed
   outside-cost budget (siblings temporarily cost zero).
2. Forward inference seeds observational terms, waits for premises,
   combines their formula classes, and transports admitted classes to
   equivalent queries. Transported classes are themselves eligible for
   transport, without inserting probability steps.

HiProF on the complete DAG deduplicates each query's subproof classes.
Admitted classes, not just query identities, drive inference. An improved
cost for an existing class reopens its dependent combinations; this also
prevents inadmissible custom heuristics from losing affordable combinations.
Certificate replay remains independent of HiProF. Exact class membership
and completeness claims are relative to its equality checks.

**Why composition matters.** The former `layer/direct` recursion transported
only directly derived formulas and missed valid compositions. On napkin,
one can transport the kernel `K(Y | X,Z)` to `p(Y | do(X,W))` using
`p(Z | W)`, then to the target using `p(W | X)`. This third class has
probability cost 3. Independent hand-built certificates and a small
brute-force proof-tree oracle now test the agenda, rather than relying
only on comparing two engines with the same missing move.

**Termination.** Free cycles that produce no new classes terminate through
class deduplication. Cycles producing genuinely different weighting kernels
need not terminate: a finite probability-cost bound can contain infinitely
many classes. `limit` requests a prefix. `max_expansions` defaults to 10000
and counts cataloged queries plus popped agenda items; reaching it raises
`SearchLimitExceeded`, carrying partial `derivations` and `expanded` count.
It never returns a silently truncated exhaustive result. None disables the
guard. This is an expansion budget, not a hard memory or wall-time limit.
Positive transport cost makes a finite bound enumerate finitely many
proofs; large bounds can still exceed the budget or available memory.

### A* configuration

`astar.derive` configures the shared agenda with query heuristics and
optional tie-breakers. A query heuristic lower-bounds its cheapest proof
(inside cost). Shortest backward paths charge rule costs plus heuristic
bounds for sibling premises, giving an outside-context lower bound for
subproofs. Agenda priority is committed cost plus unresolved-premise bounds
plus outside cost. Admissible inside bounds preserve minimum-cost discovery
without requiring consistency. Zero estimates give the uniform-cost
relaxation of contexts. Custom overestimates can change discovery order;
actual-cost checks alone enforce the bound.

`heuristics.py` supplies `zero`, `free_proof_bound`, `lookahead_bound`, and
`max_of`. Configure factories with the search's graph and costs. They relax
transport costs to zero even when positive; this weakens but does not
invalidate their lower bounds. A `tiebreaker` callable or iterable receives
a `SearchState` for a local inference, not the whole target proof tree.
Its pending premises use provisional kernel placeholders; completed
candidates use actual simplified expressions. Scores sum after the A*
priority, with insertion order last. They do not guarantee globally smallest
formulas. The previous partial-root-tree traversal and discovery tie order
have changed; cost defaults and saved Derivation representations remain
compatible, while previously missed classes are intentionally added.

### Arguments

```python
derive(graph, treatments, outcomes, bound, *, costs=None,
       limit=None, max_expansions=10000) -> list[Derivation]
```

The graph and variables follow ID's parsing conventions. `bound` is total
rule cost; `costs` accepts `marginalization`, `conditioning`, `chain rule`,
and `transport`. The first three default to 1 and must be positive integers;
transport defaults to 0 and must be nonnegative. `base` is always free.
Invalid arguments raise `ValueError`.

A successful no-limit return means the agenda exhausted its bound. A class
limit requests only a prefix. `SearchLimitExceeded` explicitly reports an
incomplete run and exposes replayable partial results. An empty bounded
result, even exhausted, does not establish non-identifiability. Typical
first-formula bounds with free transport remain 2 for adjustment, 3 for
napkin, and 4 for front-door. With transport cost 1, napkin has two classes
through cost 6 and four through cost 7.

Both experiment runners record `search_semantics=composed_transport_v1`
and `status` (`exhausted`, `class_limit`, `expansion_limit`). They save and
verify partial results when the expansion budget is reached. Existing JSON
files without that marker used the incomplete single-transport search;
previously reported napkin class counts are not exhaustive benchmarks for
the corrected search.

### The output: `Derivation`

A [`Derivation`](src/formula_search/derivation.py#L49) is a frozen tree of
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
| `cost(costs)` | Total rule cost, including configured transport costs |
| `verify(graph)` | Replays the tree. It checks that every probability step is among `rules(graph, query)`, that every do-calculus path is a chain of `neighbours` with the side conditions tested again, that base terms have no interventions, and that every node's expression has its query's type |

### Inspecting the output

The formulas themselves:

```python
from formula_search.derivation import derive
from formula_search.suite import CURATED

task = CURATED["napkin"]
derivations = derive(*task, bound=3, limit=3)
[d.formula for d in derivations]
# ['sum_{Z} { p(Z) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }',
#  'sum_{Z} { p(Z | X) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }',
#  "sum_{W} { p(W | X) sum_{Z} { p(Z | W) icd_{X | Z} { sum_{W'} { p(W') p(X, Y | W', Z) } } } }"]
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
# new 2 True
sorted(archive.classes[1].broken_by)
# ['W -> X', 'W -> Y', 'W <-> Z', 'X <-> Y', 'X <-> Z', 'Y <-> Z', 'Z -> Y']
```

To debug the search itself, for example to see which subqueries have many
classes, use the private `_Search` directly. Its interface may change:

```python
from formula_search.derivation import _Search
from formula_search.docalculus import Query

from itertools import islice

search = _Search(ADMG.parse(task.graph))
list(islice(search.run(task.query, 3), 3))   # explicitly request a prefix
search.expanded
len(search.components)                     # queries cataloged
sorted((len(c), str(q)) for q, c in search.classes.items())[-5:]
```

### Measured behaviour

New classes of the target at each cost, measured on 2026-10-09 on a laptop
with several runs in parallel. The times are cumulative and indicative.

The following counts describe the former single-transport search; they are
not completeness claims for composed transport.

| Task | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| front-door | 0 | 0 | 0 | 0 | 1 | 0 (cost 6: 0), all under 0.1 s |
| napkin | 0 | 0 | 0 | 2 | 0 | 4, 0.2 s in all |
| two adjustment sets | 0 | 0 | 2 | 2 | 173, 87–118 s | did not finish in 280 s |
| split product | 0 | 0 | 0 | 0 | 1, 3.7 s | 0, 187 s |

- **Recovery of B.** At cost 4 on two adjustment sets, the search finds
  ID's class, `sum_{Z1, Z2} { p(Z1) p(Z2) p(Y | T, Z1, Z2) }`. On the split
  product, its one class at cost 4 is the front-door formula over `A`. That
  formula is valid and differs from ID's longer split formula, but ID's own
  class is not reached by cost 5. The tests check recovery of B only on
  front-door, napkin, confounder and mediator, and collider and mediator.
- **Frontier.** On front-door (ii)–(iv), adjustment-rich, and Figure 15 of
  Yvernes et al., which have 6 to 9 variables, no formula appears by cost 2
  or 3. The next cost does not finish within about 200 s, but the
  front-door formula needs cost 4.
- **Class explosion.** Wherever the graph implies independences, subqueries
  gain many classes that differ only by those independences. For example,
  `p(Z1)` is also derived as `sum_{Z2} { p(Z2 | T) p(Z1 | Z2) }`, which is
  valid in G but a different function on the complete DAG. Combining them
  through the chain rule multiplies their numbers. Most of the 173 classes
  at cost 4 are such variants. Deduplication then dominates the run time,
  since it checks every candidate against every class kept for its query.
  An earlier experiment (2026-10-07) found that restricting
  marginalization, chain-rule order, or weights cuts the 173 classes only
  to about 111. The same experiment found that fingerprints, values on one
  fixed model, gave identical classes about 70× faster. Fingerprints need a
  public hook in hiprof-eq.

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
2. **A fingerprint hook in hiprof-eq,** a public `GaussianEvaluator` or
   similar, to group candidates before a fresh check
   ([A1](../formula-search-docs/IMPLEMENTATION.md#a1-evaluator)). It makes
   derivation search and A5 about 70× cheaper and is needed at any scale.
3. **What B contains**
   ([Q-016](../formula-search-docs/QUESTIONS.md#q-016-set-valued-corpus-and-benchmark-validity)):
   which weights, which fixing orders, and whether do-search is included.
4. **The false-acceptance policy**
   ([Q-015](../formula-search-docs/QUESTIONS.md#q-015-global-false-acceptance-control)),
   before large runs.

I recommend deciding 2 first, since it is cheap and every option of 1 needs
it, then using the A3 numbers below to decide 1.

### Implementation, in order

1. **Commit** the evaluator, suite, templates, and derivation search, and
   update `formula-search-docs/STATUS.md`, which still lists the evaluator as
   next.
2. **A3, the first result.** A runner that passes `identify`,
   `identify_equivalent`, and `template_formulas` through the evaluator on
   every curated task and a set of random tasks. It writes the archive
   records of
   [IMPLEMENTATION.md](../formula-search-docs/IMPLEMENTATION.md#archive-records)
   to disk, with a run log of the HiProF commit, seed, and check count, and
   reports the template classes outside B.
3. **Finish A1:** the features of each class (kernels, summed variables,
   largest conditioning set, quotients, `icd`), an `unresolved` outcome for
   constructs HiProF does not support, and fingerprint grouping once the
   hook exists.
4. **Finish A2:** label each task by `implies_constraints`, group random
   graphs into isomorphism families for the development and test split,
   bound slot sizes on larger graphs such as Sachs, and add the examples of
   Raichev et al.
5. **Finish A6:** record the frontier per task and bound, check recovery of
   B on every curated graph, diagnose the missed template classes as search
   order or missing moves, and compare with do-search. All of this depends
   on decisions 1 and 2.
6. **A5 and A4,** bottom-up enumeration and ID on weakened graphs, when
   they are taken up again.
7. **Stage B.** B1 (plain LLM sampling) needs only the evaluator. B2
   (guided derivation search) reuses `rules` and `_Search` from this
   module, ranking the edges of each query instead of expanding them
   breadth-first.
