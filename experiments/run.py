"""Run or compare formula-search methods using named JSON definitions."""

import sys
from pathlib import Path

# Also works from the per-experiment frozen source snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "record-failure":
        from _runner.failure import main
    else:
        from _runner.cli import main

    main()
