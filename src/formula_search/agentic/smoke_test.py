"""Send one plain prompt to Fireworks and print its response.
Example usage: .venv/bin/python -m formula_search.agentic.smoke_test --model accounts/fireworks/models/gpt-oss-120b  "Tell me a joke"
"""

import argparse

from formula_search.agentic.fireworks import Fireworks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt", nargs="?", default="Say hello in one short sentence."
    )
    parser.add_argument(
        "--model", help="Fireworks model ID; or FIREWORKS_MODEL"
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    print(
        Fireworks(args.model, max_tokens=args.max_tokens).complete(args.prompt)
    )


if __name__ == "__main__":
    main()
