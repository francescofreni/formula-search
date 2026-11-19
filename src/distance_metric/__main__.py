"""Support python -m distance_metric and python path/to/distance_metric/."""

import sys
from pathlib import Path


def run() -> None:
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from distance_metric.configs import CONFIG
    from distance_metric.main import main

    main(CONFIG)


if __name__ == "__main__":
    run()
