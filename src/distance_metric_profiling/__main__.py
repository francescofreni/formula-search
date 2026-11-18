"""Run the selected config via a module or directory command."""

import sys
from pathlib import Path


def run() -> None:
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from distance_metric_profiling.configs import CONFIG
    from distance_metric_profiling.main import main

    main(CONFIG)


if __name__ == "__main__":
    run()
