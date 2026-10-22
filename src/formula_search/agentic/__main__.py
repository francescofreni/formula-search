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
from urllib.error import URLError

from ..evaluator import Task
from . import Fireworks, Session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="Z -> X; X -> Y")
    parser.add_argument("--treatment", default="X")
    parser.add_argument("--outcome", default="Y")
    parser.add_argument(
        "--model", help="Fireworks model ID; or FIREWORKS_MODEL"
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional model-supported reasoning effort",
    )
    args = parser.parse_args()
    session = Session(Task(args.graph, args.treatment, args.outcome))
    model = Fireworks(
        args.model,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    def choose(prompt: str) -> str:
        response = model(prompt)
        print(response, flush=True)
        return response

    error = None
    try:
        session.run(choose, max_steps=args.max_steps)
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
