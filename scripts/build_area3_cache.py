"""Build a reversible held-out Area 3 cache for geometry processing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_real_dataset import (  # noqa: E402
    BINARY_CLASS_NAMES,
    CLASS_NAMES,
    process_area,
    to_binary_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/PSNet/PSNet5/Area_3")
    parser.add_argument("--output", default="data/PSNet/real_area3_reversible.npz")
    args = parser.parse_args()

    points, labels, center, scale = process_area(Path(args.data_root), max_per_class=0)
    np.savez_compressed(
        args.output,
        points=points,
        labels=labels,
        binary_labels=to_binary_labels(labels, CLASS_NAMES),
        class_names=CLASS_NAMES,
        binary_class_names=BINARY_CLASS_NAMES,
        scene_center=np.asarray(center, dtype=np.float64),
        scene_scale=np.asarray(scale, dtype=np.float64),
    )
    print(f"Saved {args.output} with reversible center/scale metadata")


if __name__ == "__main__":
    main()
