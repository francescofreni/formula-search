"""Run a small Fireworks-guided proof search from the command line.
Example usage:
.venv/bin/python -m formula_search.agentic \
  --model accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b \
  --graph "Z -> X; Z -> Y; X -> M; M -> Y" \
  --treatment X \
  --outcome Y \
  --max-steps 50
"""

import argparse
import json
import sys
from dataclasses import asdict
from urllib.error import URLError

from ..evaluator import Task
from . import BeamSession, DepthFirstSession, Fireworks
from .common import SearchSession
from .prompts import PROMPT_VERSION, PROMPT_VERSIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="Z -> X; X -> Y")
    parser.add_argument("--treatment", default="X")
    parser.add_argument("--outcome", default="Y")
    parser.add_argument(
        "--model", help="Fireworks model ID; or FIREWORKS_MODEL"
    )
    parser.add_argument(
        "--strategy", choices=("depth-first", "beam"), default="depth-first"
    )
    parser.add_argument(
        "--prompt-version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION
    )
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Search steps: one move or beam layer, including forced selections",
    )
    parser.add_argument("--api-timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--decision-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional model-supported reasoning effort",
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.beam_width is not None and (
        args.strategy != "beam" or args.beam_width < 1
    ):
        parser.error(
            "--beam-width requires --strategy beam and a positive value"
        )
    task = Task(args.graph, args.treatment, args.outcome)
    session: SearchSession
    if args.strategy == "beam":
        session = BeamSession(
            task,
            beam_width=args.beam_width or 3,
            prompt_version=args.prompt_version,
        )
    else:
        session = DepthFirstSession(task, prompt_version=args.prompt_version)
    model = Fireworks(
        args.model,
        timeout=args.api_timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    def choose(prompt: str) -> str:
        response = model(prompt)
        print(response, flush=True)
        return response

    error = None
    try:
        session.run(
            choose,
            max_steps=args.max_steps,
            max_decision_retries=args.decision_retries,
            on_event=lambda e: (
                print(json.dumps(e), file=sys.stderr)
                if e["event"]
                in ("decision_retry", "automatic", "incomplete_response")
                else None
            ),
        )
    except (ValueError, URLError, TimeoutError) as exc:
        error = str(exc)
        print(error, file=sys.stderr)
    status = (
        "error"
        if error
        else ("exhausted" if session.exhausted else "step_limit")
    )
    print(
        json.dumps(
            {
                "strategy": args.strategy,
                **session.prompt_template.metadata,
                "beam_width": (
                    session.beam_width
                    if isinstance(session, BeamSession)
                    else None
                ),
                "max_steps": args.max_steps,
                "decision_retries_limit": args.decision_retries,
                "decision_stats": asdict(session.decision_stats),
                "status": status,
                "error": error,
                "history": session.history,
                "pruned": session.pruned,
                "formulas": [c.formula for c in session.archive.classes],
            },
            indent=2,
        )
    )

    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
