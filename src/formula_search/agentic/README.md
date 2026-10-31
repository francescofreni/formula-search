# Agent-selected proof search

Two strategies share deterministic legal moves and archive verification:

- `DepthFirstSession`: the original model-guided depth-first traversal. The
  model chooses one move, and unchosen siblings remain for backtracking.
  `Session` remains an alias for this class for existing callers.
- `BeamSession`: expand each retained equation once, then ask the model to
  keep a fixed-width subset of all candidates across the layer. Rejected
  candidates are discarded rather than saved for backtracking.

Neither strategy uses heuristic scoring or a proof-cost bound. Costs are
recorded for analysis. Both expand the first unresolved premise of each
partial proof; chain-rule moves require solving all premises.

## Beam semantics

For width `k`, each model call returns `{"keep": [ID, ...]}` with exactly
`min(k, number_of_candidates)` distinct offered IDs. The model chooses the
survivors; its list order does not change deterministic enumeration. Even
when all candidates fit, selection still goes through the model. Invalid
selections leave the frontier and archive unchanged.

Each layer applies one legal move per retained parent, generating all its
successors before selection. Complete plans offer final proof/transport-weight
variants as `submit` candidates in the following layer. Only selected submit
candidates are replayed and checked by `Archive.submit`. They enter the
archive and leave the beam; the other selected candidates form the next
frontier. Thus the frontier can contain fewer than `k` partial equations.
Completed candidates that were not selected are not submitted automatically.

Beam width is global, not per parent. Duplicate pruning retains one cheapest
representative per candidate state and suppresses states already reached at
an equal or lower cost. Rejected candidates are not marked visited, so another
route can rediscover them. This is conservative structural pruning; final
formula classes still use the archive's complete-DAG equivalence checks.

`max_steps` means model calls: one move for depth-first, one whole layer for
beam. Equal decision counts are not equal expansion budgets. Width caps
survivors, not candidate count, prompt length, memory or elapsed time. Beam
search is incomplete: `exhausted` means its retained frontier is empty, not
that every possible formula was explored. Both strategies keep state in
memory and can resume via another `run()` call.

```bash
.venv/bin/python -m formula_search.agentic \
  --strategy beam --beam-width 3 --max-steps 20 \
  --model accounts/fireworks/models/glm-5p3-flash \
  --reasoning-effort low \
  --graph "Z -> X; Z -> Y; X -> M; M -> Y" \
  --treatment X --outcome Y
```

Use `--strategy depth-first` for the old traversal (also the default).
`--beam-width` is valid only with beam search and defaults to 3. CLI output
records the strategy, width, decision limit, selections, discarded candidates,
formulas and termination status. Beam history has one record per layer;
each selected entry records its parent equation, result, cost and rule.

```python
from formula_search.agentic import BeamSession, DepthFirstSession, Fireworks
from formula_search.evaluator import Task

task = Task("Z -> X; X -> Y", "X", "Y")
session = BeamSession(task, beam_width=3)
# Or: session = DepthFirstSession(task)
archive = session.run(Fireworks(), max_steps=20)
```

The comparison runner uses a nominal 100-action budget: depth-first gets 100
calls; beam gets `100 // beam_width` calls. This accounting belongs to the
experiment runner; the session's `max_steps` API still counts model calls.

## Engineering layout

- `common.py`: session interface, deterministic move generation, partial
  equation rendering, and structural state identity.
- `depth_first.py`: stack traversal and automatic backtracking.
- `beam.py`: frontier expansion, layer deduplication, selection and discard.
- `fireworks.py`: HTTP transport and decision schemas for either strategy.
- `__main__.py`: CLI configuration and result reporting.
- `loop.py`: compatibility import for the old `Session` path.

The sections below describe the original depth-first interface unless they
explicitly mention beam search.

## Fireworks

No extra SDK is required. The adapter uses Fireworks' Chat Completions API
with a JSON schema whose ID enum contains only the offered options. Beam
selections additionally specify array length and distinct IDs, checked again
locally before execution:
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

No explicit output-token cap is sent by default. Provider defaults and model
limits still apply. `--max-tokens` remains an optional override for callers
that explicitly want a cap. The search CLI also accepts
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
  --reasoning-effort none --max-steps 20
```

Disabling thinking still leaves branch selection entirely to the model.
Fireworks documents this control at
https://docs.fireworks.ai/tools-sdks/nim-compatibility .


## Shared discovery objective

Both strategies receive the same objective, defined once in `common.py`:

> Your goal is to discover as many distinct valid identifying formulas as
> possible. Use the known formulas to avoid repeatedly pursuing the same
> result. Favor promising unexplored derivation routes, while completing
> proofs so their formulas can be verified. Choose only from the offered
> legal moves.

Strategy-specific instructions then specify one move for depth-first or a
fixed-width selection for beam. Previously saved prompts remain the record
of what older runs actually received; source edits do not update running
processes. New runs archive the updated prompts and source snapshot.

## What the model sees

Depth-first prompts show the original target, graph, current partial equation,
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

Beam prompts show the target, graph, width, layer, known formulas, and every
candidate with its parent equation, branch path, rule and resulting equation.

Backtracking in depth-first search restores the active branch's path. Completed or abandoned
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

### Temporary API failures

Fireworks requests default to a 300-second timeout and five retries (six attempts
maximum). Timeouts, connection failures, HTTP 408/429 and HTTP 5xx are retried
with exponential backoff (2 seconds doubling to 60 seconds), independent jitter,
and any longer Retry-After delay. Other HTTP errors and invalid model responses
fail immediately. Retries repeat the same decision; they do not consume search
steps or modify the frontier. A timed-out request may still incur provider usage.

Use `--api-timeout` and `--max-retries` in the CLI or comparison runner; setting
`--max-retries 0` disables retries. Retry, recovery, and exhaustion events log to
stderr. Experiments additionally save `retries.jsonl` with decision numbers and
append readable events to `run.log`. Settings are included in replay manifests.
The agentic-only Slurm fleet allows two concurrent jobs.
