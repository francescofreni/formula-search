# Experiments

Use one runner for a single experiment or a comparison. Each invocation
creates a new named result folder; earlier results stay in place.
Run commands from the repository root using the installed `.venv`.

## Choose what to run

```bash
# Show available experiment definitions; inspect cases without running them.
.venv/bin/python experiments/run.py list
.venv/bin/python experiments/run.py plan comparison

# One local A* experiment (use a compute node).
.venv/bin/python experiments/run.py run single

# Submit the complete comparison as a Slurm array.
.venv/bin/python experiments/run.py submit comparison --name baseline-october

# Submit just one method, graph, and repeat from that same definition.
.venv/bin/python experiments/run.py submit comparison \
  --task napkin --strategy beam --repeat 1 --name napkin-beam
```

`--task` and `--strategy` can each be repeated. Strategy selection accepts
a configured variant's name or its method (`derivation`, `astar`,
`depth-first`, `beam`). `--repeat` is a one-based independent run number,
not a search step. `--account` and `--partition` are available on `submit`.
`run` executes selected cases sequentially; `submit` uses the configured
Slurm concurrency, defaulting to two workers **per submission**.
Agentic runs call Fireworks using your existing credentials.

| Definition | Contents |
| --- | --- |
| `single` | One adjustment graph, A*, 200 queue pops |
| `comparison` | Four graphs, all four methods; three repeats per agentic method |
| `agentic` | Four graphs, depth-first and beam, three repeats each |
| `prompt-comparison` | Two graphs, both agentic methods, two prompt versions |
| `astar-fleet` | Four graphs, twelve queue orderings, 200 and 1000 pops |

Copy a file from [configs/](configs/) to define another experiment. Pass
its path instead of a definition name. All method settings live there;
unknown settings are rejected. `prepare` saves the complete plan and source
snapshot without starting workers or making API calls.

## Read results

```text
experiments/results/2026-10-31_120000Z__baseline-october__a1b2c3/
  README.md                 comparison table; start here
  summary.csv               one row per graph / strategy / repeat
  experiment.json           resolved settings
  cases.json                exact cases and array indices
  manifest.json             source hashes, Python/packages, invocation
  sources.tar.gz            saved Python source, including HiProF
  source/                   code that workers actually execute
  REPLAY.md                 how to rerun the saved cases
  slurm/                    scheduler stdout and stderr
  adjustment/
    astar/
      repeat-01/
        README.md           settings, status, equations
        config.json         complete case definition
        result.json         counts, discovery times, replayable proofs
        run.log             progress and errors
```

Agentic cases also save `prompt.txt`, `decisions.jsonl` (every exact prompt,
response, and applied decision), and `retries.jsonl`. The prompt's version
and hash identify manual edits as well as named versions. `prompt_layout`
separately identifies the presentation of decision context. Beam histories
are stored once per parent, with options referring to their parent IDs. API keys are not
written to these artifacts. Each repeat has its own archive.

```bash
# Refresh the table while a run is in progress.
.venv/bin/python experiments/run.py report experiments/results/RUN_FOLDER

# Generate experiments/HISTORY.md, including links to legacy result folders.
.venv/bin/python experiments/run.py history

# Rerun a saved experiment locally, into a fresh result folder.
.venv/bin/python experiments/run.py replay experiments/results/RUN_FOLDER
```

Workers save partial results after discoveries/decisions, and refresh the
aggregate table when they finish. A hard kill can leave the last saved
status as `running`: check Slurm before interpreting it. Pending or
interrupted bounded derivation has an unavailable count until its batch
returns; that is not zero discoveries. Errors retain any saved formulas.
Local worker exits and Slurm bootstrap failures are recorded as errors while
preserving partial results. A hard kill of the entire allocation can still
prevent the error handler from running.

Generated results and logs are ignored by Git; the runner and definitions
are tracked. Earlier experiment scripts and results remain untouched.

## Interpret comparisons

All methods submit replayable derivations to the same `Archive`. Class
counts are distinct under its complete-DAG equivalence checks **within a
case**, not novel relative to other methods. Graph-specific equality does
not merge two classes. HiProF provides linear-Gaussian verification support;
it is not a general proof of equality or inequality.

Budgets measure different work:

- Derivation: maximum derivation cost; search returns a batch.
- A*: maximum priority-queue pops, including completed proof candidates.
- Depth-first: maximum applied moves, including forced moves.
- Beam: maximum layers, including forced selections, retaining up to `width` candidates.

If every candidate fits, beam retains all candidates without an API call.
Depth-first also skips the call for a sole option. These automatic steps still
consume the move/layer budget. `decision_retries` (default 2) allows up to two
corrective retries for invalid selections or exhausted response budgets on
the same state. It does not change reasoning effort or token limits. Failed
attempts do not consume moves/layers. Tables and CSVs separate model calls,
automatic steps, and corrective retries; HTTP retries have their own counter.

These are controlled comparisons with explicit budgets, not equal-compute
comparisons. A beam layer can expand several proofs. Lookahead and changing
queue scores also cost work beyond a pop. Exhaustion can occur before a
cap. Slurm still imposes its configured allocation time limit; an API
request timeout and retries limit individual requests, not search depth.

A* profiles record both ordering fields and their rationale. In profile
notation, `g` is cost so far, `h` is a depth-two lookahead lower bound on
remaining cost, and `f` is their sum. `size` counts expression nodes.
`N` is the fraction of distinct terms absent from discovered formulas;
`E` is `1 / (1 + visits to the next unresolved query)`; `D` averages `N`
and `E`. Smaller scores come first, so negative novelty/exploration scores
favor larger values. Secondary scores break primary-score ties. Novelty
is a guide, not a guarantee of a new formula class. Profiles that promote
novelty above cost do not promise minimum-cost discovery order.

## Reproduce a run

Workers execute the source snapshot captured at preparation time, so later
checkout changes do not affect queued cases. The manifest records file
hashes and installed dependency versions. Replay needs those dependencies
and credentials supplied separately; it does not bundle a Python runtime.
The verifier seed and worker hash seed are fixed. Hosted models may still
vary between repeats even at temperature zero, and model names can refer
to changing provider deployments. Keep the saved decisions for auditing.
A started case cannot be overwritten: replay creates a fresh cohort alongside
the original run by default (`--output-root` overrides this). Its manifest
preserves the original source commit and records the parent run and source
hash, even if the checkout has moved to another commit.
