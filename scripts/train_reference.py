"""Prepare or execute the pinned ResPointNet++ Phase-0 command.

Execution is opt-in because the PSNet5 download, CUDA compiler, and GPU host
are external resources. This script never stores or reads SSH credentials.
"""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepsegregation.reference import inspect_reference, training_command


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="vendor/ResPointNet2")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.reference).resolve()
    status = inspect_reference(root)
    if not status.ready_for_training:
        raise SystemExit(f"reference tree is incomplete: {status}")
    command = training_command(root, data_root=args.data_root, num_gpus=args.gpus, smoke=args.smoke)
    print("cd", root)
    print(" ".join(command))
    if args.execute:
        return subprocess.call(command, cwd=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
