"""Evaluate a binary checkpoint over every point in a PSNet5 validation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepsegregation.metrics import segmentation_metrics
from deepsegregation.model import build_pointnet2_ssg, build_point_mlp


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points - center, axis=1).max()
    return (points - center) / max(float(scale), 1e-6)


def evaluate_checkpoint(checkpoint, cache, device, model_name, cell_size, max_points):
    import torch

    data = np.load(cache)
    points = data["points"].astype(np.float32)
    labels = data["binary_labels"].astype(np.int64)
    model = (
        build_pointnet2_ssg(3, 2) if model_name == "pointnet2_ssg"
        else build_point_mlp(3, 2)
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    origin = points.min(axis=0)
    cell_ids = np.floor((points - origin) / cell_size).astype(np.int64)
    predictions = np.empty(len(points), dtype=np.int64)
    with torch.no_grad():
        for cell in np.unique(cell_ids, axis=0):
            indices = np.flatnonzero(np.all(cell_ids == cell, axis=1))
            normalized_cell = normalize_points(points[indices])
            for start in range(0, len(indices), max_points):
                chunk = indices[start:start + max_points]
                local_start = start
                local_end = start + len(chunk)
                inputs = torch.from_numpy(normalized_cell[local_start:local_end]).to(device)
                predictions[chunk] = model(inputs).argmax(dim=1).cpu().numpy()

    metrics = segmentation_metrics(labels, predictions, num_classes=2)
    metrics["point_count"] = int(len(points))
    metrics["pipe_prevalence"] = float(np.mean(labels == 1))
    metrics["pipe_precision"] = float(
        metrics["confusion_matrix"][1][1] /
        max(1, metrics["confusion_matrix"][0][1] + metrics["confusion_matrix"][1][1])
    )
    metrics["pipe_recall"] = float(metrics["per_class_recall"][1])
    metrics["pipe_f1"] = float(
        2 * metrics["pipe_precision"] * metrics["pipe_recall"] /
        max(1e-12, metrics["pipe_precision"] + metrics["pipe_recall"])
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", default="data/PSNet/real_val.npz")
    parser.add_argument("--model", choices=["point_mlp", "pointnet2_ssg"], default="pointnet2_ssg")
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    import torch

    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metrics = evaluate_checkpoint(
        checkpoint, args.cache, device, args.model, args.cell_size, args.max_points
    )
    payload = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "cell_size": args.cell_size,
        "device": str(device),
        "metrics": metrics,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
