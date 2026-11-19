"""Send one plain prompt to a model and print its response.
Example usage: .venv/bin/python -m formula_search.agentic.smoke_test --model accounts/fireworks/models/gpt-oss-120b  "Tell me a joke"
"""

import argparse

from formula_search.agentic.providers import PROVIDERS, create_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt", nargs="?", default="Say hello in one short sentence."
    )
    parser.add_argument("--provider", choices=PROVIDERS, default="fireworks")
    parser.add_argument(
        "--model", help="Model ID or provider model environment variable"
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    print(
        create_model(
            args.provider, args.model, max_tokens=args.max_tokens
        ).complete(args.prompt)
    )


if __name__ == "__main__":
    main()
