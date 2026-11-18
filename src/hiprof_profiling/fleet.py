"""Run all named presets in configs.py, in their declared order."""

import sys
from pathlib import Path


def main() -> None:
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from hiprof_profiling import Config, configs
    from hiprof_profiling.main import main as profile

    presets = [
        (name, config)
        for name, config in vars(configs).items()
        if name != "CONFIG" and isinstance(config, Config)
    ]
    for i, (name, config) in enumerate(presets, start=1):
        print(f"\n[{i}/{len(presets)}] {name}", flush=True)
        profile(config)
    print(f"\nCompleted {len(presets)} configs.")


if __name__ == "__main__":
    main()
