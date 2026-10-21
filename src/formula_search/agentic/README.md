# Minimal agentic search

An LLM chooses A* settings, sees which formula classes were new, and chooses
again. `Session` owns one task, an `Archive`, and an in-memory history.
It accepts any callback `complete(prompt: str) -> str`; wrap your preferred
model client to return its response text. No model SDK is required here.

```python
from formula_search.agentic import Session
from formula_search.evaluator import Task

session = Session(
    Task("Z -> X; X -> Y", "X", "Y"),
    max_bound=2,
    max_limit=5,
)

# A runnable scripted controller; replace with your model callback.
responses = iter([
    '{"action":"search","bound":0,"limit":1}',
    '{"action":"search","bound":1,"limit":5,'
    '"heuristic":"free","tiebreaker":"term_count"}',
    '{"action":"stop"}',
])
archive = session.run(lambda prompt: next(responses), max_rounds=3)
for known in archive.classes:
    print(known.formula)
# p(Y | X)
# sum_{Z} { p(Z) p(Y | X, Z) }
```

For a manual or external tool loop, send `session.prompt()` to the model,
then pass its JSON response to `session.step(response)`. The returned record
contains the action, stopping status, number of new/total classes, and each
formula's archive verdict. `session.history` is JSON-serializable. Actual
proof objects are retained as sources in `session.archive.log`.

The prompt specifies two actions: `search` and `stop`. Search requires a
nonnegative integer `bound` and positive integer `limit`. Optional guidance
is `heuristic="zero"` or `"free"`, and `tiebreaker="none"`,
`"expression_size"`, or `"term_count"`. Session caps are caller-controlled.
Invalid responses raise `ValueError`; callback and engine failures propagate.
There are no automatic retries or execution of model-generated code.

Each search starts fresh. Its limit counts all classes returned, including
ones already archived. Increase the limit or change the configuration to
explore further. `class_limit` means the returned prefix hit that limit;
`exhausted` means only that this bounded engine search finished. Reaching
`max_rounds` leaves the session available for further calls. Neither stopping
nor finding nothing establishes completeness or non-identifiability.

This first version guides whole searches, not individual branches. It adds
no rules to the underlying engine and inherits its search-space limitations.
The engine constructs derivations; `Archive.submit` replays each proof and
checks validity and class equality with HiProF. Equality is tested on the
complete DAG, with HiProF's existing linear-Gaussian limitations.

There is no frontier resumption, disk persistence, per-expansion budget, or
wall-time limit. Small bounds can still take substantial work. Model calls
happen once per round; a real callback determines credentials, model,
sampling settings, and any API charges. Store those settings externally if
you need reproducible experiments.
