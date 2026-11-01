"""Structured exploratory instructions; preserve this version once evaluated."""

DEPTH_FIRST = """
Goal:
Discover as many distinct valid identifying formulas as possible.

What counts as distinct:
All valid formulas identify the same effect in the task graph. Distinctness
means different functions of an otherwise unconstrained observational
distribution, not different causal effects. The program checks validity and
whether a completed formula is distinct. Longer expressions and unfamiliar
terms alone do not establish novelty.

Exploration:
- Compare the offered moves with known formulas and the current branch.
- Look for alternative adjustment variables, mediator decompositions,
  probability factors, conditioning sets, and ways of splitting or averaging
  expressions.
- When a similar formula is already known, consider which factors, conditioning
  sets, or averaging variables the proposed route could change. Use this
  contrast as a guide, not as proof of novelty.
- If no formulas are known, seek a first completed proof while considering
  alternative routes.

Completion:
- Choose base moves when they finish a useful route, and probability moves
  when they open a concrete alternative.
- Consider whether an exploratory move offers a plausible way to resolve a
  remaining expression, rather than merely expanding the proof.
- These are soft preferences. Useful routes may initially resemble known
  formulas, introduce more subproblems, or require extra steps. You need not
  know the complete solution before choosing a move.
- A submit move records a completed formula. Resolving its final subproblem
  alone does not archive it. Select promising submit options to record
  discoveries rather than leaving completed proofs unsubmitted.

Depth-first strategy:
- Use the current branch's history to avoid repeatedly undoing earlier progress.
- When several moves seem promising, prefer a route different from previously
  completed formulas.
- Choose a move for the first unresolved subproblem. You cannot choose which
  subproblem to expand.
- There is no backtrack action. The program retains unchosen alternatives and
  backtracks automatically when the active branch has no moves.

Legal moves and notation:
Choose only from the offered legal moves. All offered moves are legal.
Costs are recorded only; there is no proof-cost bound.
UNRESOLVED marks a probability expression still to derive.
Marginalize[V] sums out V; Condition[V] conditions on V.
Transport converts equivalent queries; its averaging weights remain deferred
until a submit option. Equations containing these operators are proof templates.
All unresolved terms must eventually be resolved.

Output format:
Return only {"move": INTEGER_ID}.
Use the numeric id field of an offered option, not its position in the list.
Return no commentary, Markdown fences, or additional fields.
""".strip()

BEAM = """
Goal:
Discover as many distinct valid identifying formulas as possible.

What counts as distinct:
All valid formulas identify the same effect in the task graph. Distinctness
means different functions of an otherwise unconstrained observational
distribution, not different causal effects. The program checks validity and
whether a completed formula is distinct. Longer expressions and unfamiliar
terms alone do not establish novelty.

Exploration:
- Compare the offered moves with known formulas and their branch paths.
- Look for alternative adjustment variables, mediator decompositions,
  probability factors, conditioning sets, and ways of splitting or averaging
  expressions.
- When a similar formula is already known, consider which factors, conditioning
  sets, or averaging variables the proposed route could change. Use this
  contrast as a guide, not as proof of novelty.
- If no formulas are known, seek a first completed proof while considering
  alternative routes.

Completion:
- Choose base moves when they finish a useful route, and probability moves
  when they open a concrete alternative.
- Consider whether an exploratory move offers a plausible way to resolve a
  remaining expression, rather than merely expanding the proof.
- These are soft preferences. Useful routes may initially resemble known
  formulas, introduce more subproblems, or require extra steps. You need not
  know the complete solution before choosing a move.
- A submit move records a completed formula. Resolving its final subproblem
  alone does not archive it.

Beam strategy:
- Choose a complementary set, not just individually attractive candidates.
- Avoid filling the beam with minor variations of one route when different
  routes are available. Different parents alone do not imply different routes;
  compare their equations and branch paths.
- Prefer coverage of different factorizations, conditioning sets, and averaging
  variables when these suggest distinct routes with plausible completions.
- Balance candidates close to completion with candidates exploring different
  decompositions. When width permits, include promising submit options
  alongside unfinished routes.
- A promising route may merit multiple slots. Do not impose a fixed quota per
  parent or rule.
- Select across all parents together. Unselected candidates are discarded.
- Selected submit options are verified and archived. They do not occupy slots
  in the next frontier.

Legal moves and notation:
Choose only from the offered legal moves. All offered moves are legal.
Costs are recorded, not a budget or ranking.
UNRESOLVED marks a probability expression still to derive.
Marginalize sums out variables; Condition conditions a probability expression.
Transport converts equivalent queries with final weights deferred until submit.
All unresolved terms must eventually be resolved.

Output format:
Return only {"keep": [INTEGER_ID, ...]}.
Select exactly selection_count distinct IDs from options.
Use each option's numeric id field, not its position in the list.
Return no commentary, Markdown fences, or additional fields.
""".strip()
