"""Calibrate DBSCAN settings on a validation scene only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepsegregation.clustering import dbscan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.cache)
    points = data["points"].astype(np.float64) * float(data["scene_scale"]) + data["scene_center"]
    pred = np.load(args.predictions)
    pipe = points[pred == 1]
    rng = np.random.default_rng(42)
    pipe = pipe[rng.choice(len(pipe), min(300000, len(pipe)), replace=False)]
    results = []
    for eps in (0.02, 0.03, 0.05, 0.08):
        for min_samples in (6, 8, 12):
            labels = dbscan(pipe, eps=eps, min_samples=min_samples)
            sizes = np.bincount(labels[labels >= 0])
            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "clusters": int(len(sizes)),
                "clusters_ge_100": int(np.sum(sizes >= 100)),
                "largest": int(sizes.max()) if len(sizes) else 0,
            })
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
