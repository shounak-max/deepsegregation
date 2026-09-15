"""Save pointwise predictions for a real PSNet5 cache."""

from __future__ import annotations

import argparse
import numpy as np
import torch

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepsegregation.model import build_pointnet2_ssg, build_point_mlp


def normalize_points(points):
    center = points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points - center, axis=1).max()
    return (points - center) / max(float(scale), 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["point_mlp", "pointnet2_ssg"], default="pointnet2_ssg")
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = np.load(args.cache)
    points = data["points"].astype(np.float32)
    model = (build_pointnet2_ssg(3, 2) if args.model == "pointnet2_ssg"
             else build_point_mlp(3, 2)).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model_state"])
    model.eval()
    origin = points.min(axis=0)
    cell_ids = np.floor((points - origin) / args.cell_size).astype(np.int64)
    predictions = np.empty(len(points), dtype=np.int64)
    with torch.no_grad():
        for cell in np.unique(cell_ids, axis=0):
            indices = np.flatnonzero(np.all(cell_ids == cell, axis=1))
            normalized = normalize_points(points[indices])
            for start in range(0, len(indices), args.max_points):
                end = min(start + args.max_points, len(indices))
                inputs = torch.from_numpy(normalized[start:end]).to(args.device)
                predictions[indices[start:end]] = model(inputs).argmax(1).cpu().numpy()
    np.save(args.output, predictions)
    print({"points": len(points), "predicted_pipe": int(np.sum(predictions == 1))})


if __name__ == "__main__":
    main()
