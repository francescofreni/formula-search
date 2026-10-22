# Agent-selected proof moves

At each node the engine enumerates legal next moves in a deterministic order.
The model receives their descriptions and returns only `{"move": ID}`. The
engine applies that move. There is no heuristic scoring or priority queue.
There is no proof-cost bound. Costs are recorded for analysis, not used to
filter legal moves. `max_steps` is the only search budget.

Traversal is depth-first: unchosen siblings stay available, and the harness
backtracks automatically after completing a proof or reaching a dead end.
The next unresolved premise is chosen in deterministic left-to-right order;
chain-rule moves require proving both premises, not choosing between them.
Even a single available move goes through the model callback. Complete proof
variants (including transport weights) are offered as `submit` moves and
are replayed and checked through `Archive.submit` when selected.

## Fireworks

No extra SDK is required. The adapter uses Fireworks' Chat Completions API
with a JSON schema whose move-ID enum contains only the offered options:
https://docs.fireworks.ai/structured-responses/structured-response-formatting

Set `FIREWORKS_MODEL` to the exact model/deployment ID from your Fireworks
account. Use a model supporting structured outputs. The key is read from
`FIREWORKS_API_KEY`, or from `~/.config/fireworksai/api_key` when the variable
is absent. You can also pass `model` and `api_key` explicitly.

From a compute-node shell:

```bash
export FIREWORKS_MODEL="YOUR_FIREWORKS_MODEL_ID"
.venv/bin/python -m formula_search.agentic --max-steps 20
```

The default task is `Z -> X; X -> Y`, targeting the effect of X on Y. Override
it with `--graph`, `--treatment`, and `--outcome`. The command prints decisions,
then the history, formulas, and `exhausted` or `step_limit` status.

```python
from formula_search.agentic import Fireworks, Session
from formula_search.evaluator import Task

session = Session(Task("Z -> X; X -> Y", "X", "Y"))
model = Fireworks()  # FIREWORKS_MODEL and your configured key
archive = session.run(model, max_steps=20)

# Continue the same frontier if the first decision budget ran out.
if not session.exhausted:
    session.run(model, max_steps=20)

for known in archive.classes:
    print(known.formula)
```

Each step makes one API request (no automatic retries). The adapter requests
temperature zero, but provider outputs are not guaranteed reproducible. Legal
move enumeration and execution are deterministic for the same decisions.
Model errors, truncated output, or invalid move IDs raise exceptions, with no
fallback branch selection. Invalid decisions leave the frontier unchanged.
Model calls send the graph, target, active branch path, current equation,
legal next equations, recorded costs, and known formulas to Fireworks; keys are only used in the authorization header.

## Offline or another provider

`run` still accepts any `prompt: str -> response: str` callback. Prompts are
JSON snapshots, and `prompt()` / `step(response)` support an external loop.
A deterministic first-option policy is useful for tests:

```python
import json

def first_option(prompt: str) -> str:
    return json.dumps({"move": json.loads(prompt)["options"][0]["id"]})

session.run(first_option, max_steps=20)
```

Move IDs are session-wide and never reused. History records chosen moves,
rule paths, cost, and archive verdicts. Proof objects stay in `archive.log`.
The prior `max_bound`, `max_limit`, `max_rounds`, and configuration-selection
JSON actions have been replaced by `max_steps` and move IDs. `bound` has
also been removed from Session and the CLI; old commands using it must
drop that argument.

## Scope

Moves reuse this checkout's A* `_Plan` and `_Search.expand`: an equivalent
query's shortest do-calculus path is bundled with a base or probability
rule. Individual do-calculus edges and arbitrary composed transports are
not independently selectable in this version. This inherits the engine's
search-space limitations; it does not resolve missing inference moves.

Final weight variants are enumerated deterministically from complete plans.
Generating options can be expensive. `max_steps` caps model decisions, not
memory, wall time, or option count. Search state and history are in memory;
there is no disk checkpointing. Histories/prompts grow with the search.

`exhausted` means the frontier is empty. Without a cost bound, the space
can be infinite and depth-first traversal need not visit every branch.
Reaching `max_steps` preserves the frontier in memory; a later `run` call
gets its own additional decision budget. An unfinished run does not establish
non-identifiability or discovery of every possible equation. The model's
order carries no A* minimum-cost guarantee. Certificates establish validity;
class equality uses HiProF on a complete DAG with its existing limitations.

## Test a plain model call

To check your credentials and model with one prompt, without running search:

```bash
.venv/bin/python -m formula_search.agentic.smoke_test \
  --model YOUR_FIREWORKS_MODEL_ID "Say hello in one sentence."
```

Omit `--model` if `FIREWORKS_MODEL` is set. This reads the same API key as the
search adapter, makes one request, and prints the model's text response.

## Incomplete model responses

The output budget defaults to 16,384 tokens, including model reasoning. Set
`--max-tokens` on either CLI to override it. The search CLI also accepts
`--reasoning-effort` for models supporting that parameter; no effort setting
is sent by default. These are ceilings, not target output lengths.

If Fireworks cannot produce a complete answer, the error reports its finish
reason, completion-token usage, and content length without printing reasoning
or credentials. A `length` finish means the output budget was exhausted. For models that
support it, `--reasoning-effort none` disables thinking; raising the budget
alone may just produce more reasoning without a decision. There
are no automatic paid retries and incomplete responses never select a move.
The search CLI prints its partial history and formulas with status `error`
and exits nonzero. This report is not a resumable disk checkpoint; Python
callers can retry `run` on the same Session because the frontier is intact.


For Nemotron Lightning, a minimal decision-only run is:

```bash
.venv/bin/python -m formula_search.agentic \
  --model accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b \
  --reasoning-effort none --max-tokens 512 --max-steps 20
```

Disabling thinking still leaves branch selection entirely to the model.
Fireworks documents this control at
https://docs.fireworks.ai/tools-sdks/nim-compatibility .


## What the model sees

Each prompt shows the original target, graph, current partial equation,
active branch's equations and rules, next term to expand, recorded proof
cost, and legal moves with their resulting partial equations. Previous
formulas are included to support finding new classes. The model is asked
only which offered move to apply next; it need not solve the whole proof.

`UNRESOLVED[...]` marks unfinished causal kernels, preserving interventions.
`Marginalize[V](...)` and `Condition[V](...)` operate on kernels, avoiding
ambiguous dummy-variable scope in unfinished expressions. `Transport[...]`
is explicit because its averaging weights are deferred until completion.
These are proof templates, not observational formulas ready for verification.
Submit options show the exact final rendered formula.

Backtracking restores the active branch's path. Completed or abandoned
branches do not appear in that path. The full audit log remains available
in `session.history` and CLI output, but is not sent in each model prompt.

## Duplicate-equation pruning

Pruning is enabled by default. A reached partial equation is remembered with
its lowest committed cost. Options returning to the same state at equal or
higher cost are removed before the next model call. A cheaper route is kept
and reopens the state so cheaper certificates can be retained. This
cost comparison is duplicate-state dominance, not a proof-cost cap.
Unchosen options are not marked visited merely because they were generated.

Partial-state comparison preserves unresolved query types, their expansion
order, and deferred transport endpoints/weights. It normalizes adjacent
marginalizations, so summing out A then B has the same key as B then A.
It does not replace unresolved interventions with observational terms or
assume task-graph equalities make equations interchangeable. This is a
conservative structural comparison, not a complete algebraic equivalence
test. Repeated subqueries inside different larger expressions alone do not
trigger pruning. Completed formulas use exact rendered equality here;
Archive still performs its usual complete-DAG class checks on submissions.

`session.pruned` (also printed by the CLI) records the move, equation, cost,
and earlier best cost. Set `Session(..., prune_duplicates=False)` for an
unpruned comparison. State memory, like the frontier, lasts for the Session.
