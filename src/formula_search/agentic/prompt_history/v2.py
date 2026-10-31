"""Frozen v2 instructions for controlled prompt comparisons."""

# Fold line breaks into spaces so rewrapping does not change the prompt.

DEPTH_FIRST = " ".join("""
Your goal is to discover as many distinct valid identifying formulas as
possible. Compare the offered moves with the known formulas and the current
branch. Favor alternative adjustment variables, mediator decompositions,
probability factors, conditioning sets, and ways of splitting or averaging
expressions. Explore different derivation routes rather than repeatedly
rebuilding the same formula. Do not equate longer expressions or unfamiliar

terms with genuine novelty. Complete promising proofs: choose base moves
when they finish a useful route, and probability moves when they open a
concrete alternative. A submit move records a completed formula; resolving
its final subproblem alone does not archive it. All valid formulas identify
the same effect in the task graph; distinctness means different functions
of an otherwise unconstrained observational distribution, not different

causal effects. The program determines validity and whether a completed
formula is distinct. Choose only from the offered legal moves. Use the
current branch's history to avoid repeatedly undoing earlier progress. When
several moves seem promising, prefer a route different from previously
completed formulas. You choose a move for the first unresolved subproblem,
not which subproblem to expand. There is no backtrack action: unchosen

alternatives are retained by the program, which backtracks automatically
when the active branch has no moves. Select promising submit options to
record discoveries rather than leaving completed proofs unsubmitted. Which
offered move should we apply next? Return only {"move": INTEGER_ID}. There
is no proof-cost bound; costs are recorded only. Choose one local step. All
offered moves are legal. UNRESOLVED marks a kernel still to derive.

Marginalize[V] sums out V; Condition[V] conditions on V. Transport converts
equivalent queries; its averaging weights remain deferred until a submit
option. Equations containing these operators are proof templates. All
unresolved terms must eventually be resolved.
""".split())

BEAM = " ".join("""
Your goal is to discover as many distinct valid identifying formulas as
possible. Compare the offered moves with the known formulas and the current

branch. Favor alternative adjustment variables, mediator decompositions,
probability factors, conditioning sets, and ways of splitting or averaging
expressions. Explore different derivation routes rather than repeatedly
rebuilding the same formula. Do not equate longer expressions or unfamiliar
terms with genuine novelty. Complete promising proofs: choose base moves
when they finish a useful route, and probability moves when they open a

concrete alternative. A submit move records a completed formula; resolving
its final subproblem alone does not archive it. All valid formulas identify
the same effect in the task graph; distinctness means different functions
of an otherwise unconstrained observational distribution, not different
causal effects. The program determines validity and whether a completed
formula is distinct. Choose only from the offered legal moves. Choose a

complementary set of candidates, not just individually attractive ones.
Avoid filling the beam with minor variations of one route when different
routes are available. Balance candidates close to completion with
candidates exploring different decompositions. When width permits, include
promising submit options alongside unfinished routes; submitted proofs
leave the next frontier. Different parents alone do not imply different

routes; compare their equations and branch paths. A promising route may
merit multiple slots; do not impose a fixed quota per parent or rule.
Return only {"keep": [INTEGER_ID, ...]}. Select exactly selection_count
distinct IDs from options, across all parents together. Unselected
candidates are discarded. All offered moves are legal. Costs are recorded,
not a budget or ranking. UNRESOLVED is a kernel still to derive;

Marginalize sums out variables; Condition conditions a kernel. Transport
converts equivalent queries with final weights deferred until submit.
Selected submit options are verified and archived and do not occupy a slot
in the next frontier.
""".split())
