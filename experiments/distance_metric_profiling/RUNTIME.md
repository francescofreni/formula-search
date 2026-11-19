# Runtime distributions: two versus five formulas

```bash
.venv/bin/python experiments/distance_metric_profiling/profile.py submit
```

This submits `run.slurm`: one CPU, 4 GB memory, two hours, **30 trials per
case (60 total)**. Each trial is one full 1,060-chain estimate with
`epsilon=0.05`, `delta=0.01`, `target_bound=1e-14`, automatic topological
rejection sampling, and bidirected additions enabled. Expect about 35
minutes based on the initial six-trial profile; queue time is separate.

The six-variable graph is `A -> X; A -> Y; B -> X; B -> Y; X -> M; M -> N;
N -> Y`, targeting `p(Y | do(X))`. The five formulas in `profile.py` use
joint adjustment, factorized adjustment weights, the first mediator, the
second mediator, and the full mediator chain. The pair uses the first and
fifth formulas. Archive checks validity and distinct classes before timing.

All trials run sequentially on the same worker. Seeds 0–29 are paired
across cases. Even seeds run the pair first; odd seeds run the group first.
No warmups are discarded. `elapsed_seconds` measures the full estimator,
including its base checks, graph construction, rejection sampling, HiProF
calls and averaging. It excludes imports, archive preflight and file writes.

Every submission creates a fresh `results/<timestamp>_group_runtime_<id>/`
folder. Code and HiProF grammar are frozen **before submission**, hashed,
and verified on the worker. Packages/Python versions are recorded; the
worker uses the existing virtual environment for third-party dependencies.
Generated results, snapshots and Slurm logs stay out of Git.

Use **`timings.csv`** for histograms: filter `case` (`two_formulas` or
`five_formulas`) and plot `elapsed_seconds`. A successful job has 30 rows
per case and `status.json` saying `complete`. Each completed trial also
gets its own full JSON file. Errors stop the job without silently dropping
trials. Hard scheduler termination can leave status `running`; inspect
Slurm state and the saved row count before treating any run as complete.

`config.json` holds exact parameters and formulas; `manifest.json` records
code hashes/environment; `submission.json` records the command/job ID;
`slurm/` contains output/error logs. Each trial prints its distance estimate,
uncertainty interval, runtime, and verifier calls. `timings.csv` includes
`distance`, and trial JSON files include the full uncertainty information.
`summary.json` and the final Slurm output report the mean, standard deviation,
minimum, and maximum of both runtime and distance for each case. These
repeat statistics are descriptive; they do not provide a new confidence
interval for the mean. The interval guarantee is per estimate and applies
to HiProF's linear-Gaussian validity checks, with verifier error accounted
for separately in each trial JSON.

To prepare without submitting, use `prepare` instead of `submit`.
`--trials 1` creates a separate smoke-test experiment. To execute a prepared
folder locally on a compute node:

```bash
.venv/bin/python /absolute/run/folder/source/profile.py worker /absolute/run/folder
```

Workers refuse to overwrite existing trials. Submit again for a new run.
