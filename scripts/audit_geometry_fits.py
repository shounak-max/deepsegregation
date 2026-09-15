"""Audit sampled pipe clusters and reject weak cylinder fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepsegregation.clustering import dbscan
from deepsegregation.cylinder import fit_cylinder_ransac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-points", type=int, default=300000)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--min-cluster-points", type=int, default=100)
    parser.add_argument("--fit-threshold", type=float, default=0.01)
    args = parser.parse_args()

    data = np.load(args.cache)
    points = data["points"].astype(np.float64)
    predictions = np.load(args.predictions)
    if predictions.shape != (len(points),):
        raise ValueError("prediction count does not match cache")
    if "scene_scale" not in data or "scene_center" not in data:
        raise ValueError("cache lacks reversible scene scale metadata")

    scale = float(data["scene_scale"])
    center = np.asarray(data["scene_center"], dtype=np.float64)
    original_points = points * scale + center
    pipe_indices = np.flatnonzero(predictions == 1)
    rng = np.random.default_rng(42)
    selected = rng.choice(pipe_indices, min(args.sample_points, len(pipe_indices)), replace=False)
    cluster_labels = dbscan(
        original_points[selected], eps=args.eps, min_samples=args.min_samples
    )

    fits = []
    for cluster_id in sorted(set(cluster_labels.tolist())):
        if cluster_id < 0:
            continue
        members = np.flatnonzero(cluster_labels == cluster_id)
        if len(members) < args.min_cluster_points:
            continue
        cluster = original_points[selected[members]]
        try:
            fit = fit_cylinder_ransac(
                cluster,
                distance_threshold=args.fit_threshold,
                min_inliers=max(30, int(0.3 * len(cluster))),
                max_trials=200,
                random_state=42,
            )
        except (RuntimeError, ValueError):
            continue
        fits.append({
            "cluster_id": int(cluster_id),
            "points": int(len(cluster)),
            "inliers": int(fit.inlier_count),
            "inlier_ratio": float(fit.inlier_count / len(cluster)),
            "radius": float(fit.radius),
            "rmse": float(fit.rmse),
        })

    payload = {
        "cache": args.cache,
        "predictions": args.predictions,
        "pipe_points": int(len(pipe_indices)),
        "sampled_points": int(len(selected)),
        "clusters": int(len(set(cluster_labels.tolist()) - {-1})),
        "accepted_fits": fits,
        "scene_scale": scale,
        "scene_center": center.tolist(),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pipe_points": payload["pipe_points"],
        "sampled_points": payload["sampled_points"],
        "clusters": payload["clusters"],
        "accepted_fits": len(fits),
    }, indent=2))


if __name__ == "__main__":
    main()
