"""Verify the derived PSNet5 cache without loading point arrays into memory."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np


EXPECTED_ORIGINAL = {0, 1, 2, 3, 4}
EXPECTED_BINARY = {0, 1}
REQUIRED_FIELDS = {
    "points",
    "labels",
    "class_names",
    "binary_labels",
    "binary_class_names",
}


def verify(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"cache file is missing or empty: {path}")
    with np.load(path, allow_pickle=False) as data:
        fields = set(data.files)
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise RuntimeError(f"{path} is missing fields: {sorted(missing)}")
        labels = np.unique(data["labels"])
        binary = np.unique(data["binary_labels"])
        if set(labels.tolist()) - EXPECTED_ORIGINAL:
            raise RuntimeError(f"{path} has invalid original labels: {labels.tolist()}")
        if set(binary.tolist()) != EXPECTED_BINARY:
            raise RuntimeError(f"{path} must contain binary labels 0 and 1: {binary.tolist()}")
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "points": int(len(data["points"])),
            "fields": sorted(fields),
            "original_label_values": labels.tolist(),
            "binary_label_values": binary.tolist(),
            "binary_counts": {
                str(value): int((data["binary_labels"] == value).sum())
                for value in sorted(EXPECTED_BINARY)
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PSNet5 derived cache files")
    parser.add_argument("--root", default="data/PSNet")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    results = [verify(Path(args.root) / name) for name in ("real_train.npz", "real_val.npz")]
    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
