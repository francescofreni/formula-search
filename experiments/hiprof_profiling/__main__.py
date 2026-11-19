"""Support both module execution and `python path/to/hiprof_profiling`."""

import sys
from pathlib import Path


def run() -> None:
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from experiments.hiprof_profiling.configs import CONFIG
    from experiments.hiprof_profiling.main import main

    main(CONFIG)


if __name__ == "__main__":
    run()
