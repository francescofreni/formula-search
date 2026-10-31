"""Frozen v1 instructions for controlled prompt comparisons."""

# Fold line breaks into spaces so rewrapping does not change the prompt.

DEPTH_FIRST = " ".join("""
Your goal is to discover as many distinct valid identifying formulas as
possible. Use the known formulas to avoid repeatedly pursuing the same
result. Favor promising unexplored derivation routes, while completing
proofs so their formulas can be verified. Choose only from the offered
legal moves. Which offered move should we apply next? Return only {"move":
INTEGER_ID}. There is no proof-cost bound; costs are recorded only. Choose

one local step. All offered moves are legal. UNRESOLVED marks a kernel
still to derive. Marginalize[V] sums out V; Condition[V] conditions on V.
Transport converts equivalent queries; its averaging weights remain
deferred until a submit option. Equations containing these operators are
proof templates. All unresolved terms must eventually be resolved.
""".split())

BEAM = " ".join("""
Your goal is to discover as many distinct valid identifying formulas as

possible. Use the known formulas to avoid repeatedly pursuing the same
result. Favor promising unexplored derivation routes, while completing
proofs so their formulas can be verified. Choose only from the offered
legal moves. Return only {"keep": [INTEGER_ID, ...]}. Select exactly
selection_count distinct IDs from options, across all parents together.
Unselected candidates are discarded. All offered moves are legal. Costs are

recorded, not a budget or ranking. UNRESOLVED is a kernel still to derive;
Marginalize sums out variables; Condition conditions a kernel. Transport
converts equivalent queries with final weights deferred until submit.
Selected submit options are verified and archived and do not occupy a slot
in the next frontier.
""".split())
