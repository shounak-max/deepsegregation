"""Generate the Phase-1 synthetic dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepsegregation.synthetic import write_synthetic_dataset


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="data/pipes")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(write_synthetic_dataset(args.output, args.count, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
