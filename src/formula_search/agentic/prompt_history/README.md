# Prompt history

These are frozen **complete instruction strings** for both search strategies.
They include the discovery objective, strategy guidance, and response-format
instructions. Graphs, branch paths, known formulas and legal move options remain
runtime context and are recorded in each actual decision prompt.

| Version | Source | What changed |
| --- | --- | --- |
| `discovery-v1` | `v1.py` | Original objective: seek distinct formulas and finish proofs. Recovered from commit `f28b595`; that commit used the same text as the earlier agentic experiments. |
| `exploratory-v2` | `v2.py` | Concrete alternative derivations, complementary beam routes, and explicit completion/submission guidance. Preserved from the reviewed implementation in this conversation; not previously committed or evaluated live. |
| `exploratory-v3` | `v3.py` | Adds contrasts with known formulas and plausible routes to completion, as soft preferences. No live performance claim. |

| `structured-v4` | `v4.py` | Readable Goal, Exploration, Completion, Strategy, and Output sections; explicit ID/output instructions. Current default. |

## Use

Python: `DepthFirstSession(task, prompt_version="discovery-v1")` or
`BeamSession(task, beam_width=10, prompt_version="exploratory-v3")`.
Both the CLI and comparison runner accept `--prompt-version`.

From the repository root, submit otherwise identical agentic fleets:

```bash
bash experiments/agentic/submit_fleet.sh --prompt-version discovery-v1
bash experiments/agentic/submit_fleet.sh --prompt-version exploratory-v3
```

Each submission has a separate results folder. Run cohorts sequentially if you
want to retain a total of two concurrent API jobs; the Slurm limit is per array.
These commands submit paid model calls; merely selecting a version in Python
makes no API request.

Each result bundle saves `prompt.txt`, its version and SHA-256 in configuration,
the exact requests in `decisions.jsonl`, and the version in its replay command.
CSV output from `experiments/comparisons/summarize.py` includes version and hash;
old bundles without these fields remain unlabelled rather than guessed.
The hash identifies static instructions only, not the graph or response schema
implemented by the provider adapter. Source snapshots preserve those too.

## Compare fairly

Use the same code checkout, model, graph/query, search strategy, beam width,
budgets, provider settings and repeats. Report verified classes, actual steps,
runtime and failures per repeat. Keep timeout/rate-limit failures visible;
do not compare a partial failed run as though it used the full budget.
Hosted model responses are not bitwise reproducible. Switching to old wording
on new code is a controlled prompt comparison, not a recreation of every aspect
of an old experiment. Use its source snapshot for historical replay.

## Add a version

Prompt bodies use triple-quoted text. In v4, blank lines and headings are sent
to the model as written; older versions normalize source wrapping to preserve
the original text and hashes.

Do not edit an existing version's strings. Copy a version file, change the copy,
and register a new name in `prompts.py`. Update this table and the frozen hashes
in `tests/test_agentic_prompts.py`. Change the default separately and explicitly.
The hash regression test detects accidental edits to historical instructions.
