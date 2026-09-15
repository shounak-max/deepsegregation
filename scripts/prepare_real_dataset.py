"""Preprocess the original PSNet5 areas into derived training tensors.

The raw PSNet5 distribution is expected under ``data/PSNet/PSNet5`` and must
remain outside version control. This utility reads the full annotation files
by default; use ``--points-per-class`` only for a deliberate smoke run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CLASS_NAMES = ["ibeam", "pipe", "pump", "rectangularbeam", "tank"]
NAME_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}
BINARY_CLASS_NAMES = ["background", "pipe"]


def to_binary_labels(labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Map original PSNet5 labels to pipe/background without silent fallback."""
    names = [str(name).lower() for name in class_names]
    if names != CLASS_NAMES:
        raise ValueError(
            f"unexpected PSNet5 class order: {names}; expected {CLASS_NAMES}"
        )
    values = np.asarray(labels)
    if np.any((values < 0) | (values >= len(names))):
        raise ValueError("labels contain values outside the PSNet5 class range")
    return (values == NAME_TO_LABEL["pipe"]).astype(np.int64)


def sample_points_from_file(filepath: str, max_points: int | None = None) -> np.ndarray:
    """Read all points, or a deterministic uniform cap for a smoke run."""
    file_size = os.path.getsize(filepath)
    # Estimate line count (~30-35 bytes per line)
    est_lines = max(1, file_size // 32)
    stride = 1 if max_points is None or max_points <= 0 else max(1, est_lines // max_points)

    points = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if idx % stride == 0:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
                if max_points is not None and max_points > 0 and len(points) >= max_points:
                    break
    return np.array(points, dtype=np.float32)


def process_area(area_path: Path, max_per_class: int = 50000):
    all_points = []
    all_labels = []
    anno_files = list(area_path.glob("Room_*/Annotations/*.txt"))
    for f in anno_files:
        cls_name = f.name.split("_")[0].lower()
        if cls_name not in NAME_TO_LABEL:
            continue
        label = NAME_TO_LABEL[cls_name]
        pts = sample_points_from_file(str(f), max_points=max_per_class)
        if len(pts) > 0:
            all_points.append(pts)
            all_labels.append(np.full(len(pts), label, dtype=np.int64))
            print(f"    Loaded {cls_name:15s}: {len(pts):,d} points")

    if not all_points:
        raise RuntimeError(f"No points loaded from {area_path}")

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    # Normalize to zero center and unit scale
    center = points.mean(axis=0)
    scale = np.max(np.linalg.norm(points - center, axis=1)) + 1e-6
    points = (points - center) / scale
    return points.astype(np.float32), labels.astype(np.int64), center, scale


def build_real_splits(dataset_root: Path, output_dir: Path, points_per_class: int = 50000):
    output_dir.mkdir(parents=True, exist_ok=True)
    psnet_dir = dataset_root

    # Keep Area_3 untouched until final held-out evaluation.
    train_areas = ["Area_1", "Area_2"]
    val_areas = ["Area_4"]

    print("=" * 60)
    print(f"Building Real Dataset Splits from {psnet_dir}")
    print("=" * 60)

    train_pts, train_lbls = [], []
    for area in train_areas:
        a_dir = psnet_dir / area
        if not a_dir.exists():
            print(f"Warning: {a_dir} not found, skipping...")
            continue
        print(f"Processing Train Area: {area}...")
        pts, lbls, _, _ = process_area(a_dir, max_per_class=points_per_class)
        train_pts.append(pts)
        train_lbls.append(lbls)

    if not train_pts:
        raise RuntimeError("No training areas were found in the original PSNet5 dataset")
    train_points = np.vstack(train_pts)
    train_labels = np.concatenate(train_lbls)
    train_file = output_dir / "real_train.npz"
    np.savez_compressed(
        train_file,
        points=train_points,
        labels=train_labels,
        class_names=CLASS_NAMES,
        binary_labels=to_binary_labels(train_labels, CLASS_NAMES),
        binary_class_names=BINARY_CLASS_NAMES,
    )
    print(f">> Saved {train_file}: {len(train_points):,d} points across {len(CLASS_NAMES)} classes")

    val_pts, val_lbls = [], []
    for area in val_areas:
        a_dir = psnet_dir / area
        if not a_dir.exists():
            print(f"Warning: {a_dir} not found, skipping...")
            continue
        print(f"Processing Val Area: {area}...")
        pts, lbls, _, _ = process_area(a_dir, max_per_class=points_per_class)
        val_pts.append(pts)
        val_lbls.append(lbls)

    if not val_pts:
        raise RuntimeError("No validation area was found in the original PSNet5 dataset")
    val_points = np.vstack(val_pts)
    val_labels = np.concatenate(val_lbls)
    val_file = output_dir / "real_val.npz"
    np.savez_compressed(
        val_file,
        points=val_points,
        labels=val_labels,
        class_names=CLASS_NAMES,
        binary_labels=to_binary_labels(val_labels, CLASS_NAMES),
        binary_class_names=BINARY_CLASS_NAMES,
    )
    print(f">> Saved {val_file}: {len(val_points):,d} points across {len(CLASS_NAMES)} classes")
    print("=" * 60)
    print("Real dataset preparation complete!")
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="Prepare real PSNet5 dataset for training")
    parser.add_argument("--data-root", default="data/PSNet/PSNet5")
    parser.add_argument("--output-dir", default="data/PSNet")
    parser.add_argument(
        "--points-per-class",
        type=int,
        default=0,
        help="Maximum points per class; 0 uses the complete original annotation files",
    )
    args = parser.parse_args()

    build_real_splits(Path(args.data_root), Path(args.output_dir), args.points_per_class)


if __name__ == "__main__":
    main()
