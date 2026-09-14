"""Systematic sweep of normal-consistency DBSCAN parameters.

Addresses peer reviewer criticism that the 50-degree normal threshold was
validated on only one qualitative example (two touching cylinders). This
script runs a precision/recall sweep over:
  - normal_threshold_deg in [30, 40, 50, 60, 70]
  - eps in [0.01, 0.02, 0.05, 0.10]

Measures:
  - Instance count accuracy (how many physical instances are found)
  - False-merge rate (two distinct instances merged into one)
  - False-split rate (one instance fragmented into multiple clusters)

Also tests the false-split failure mode: a single pipe with simulated
weld-seam normal variation, which could be falsely split if the threshold
is too tight.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepsegregation.clustering import dbscan, estimate_normals


def generate_touching_cylinders(
    n_per_ring: int = 24,
    n_rings: int = 25,
    radius: float = 0.05,
    gap: float = 0.0,
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
) -> tuple:
    """Generate two parallel cylinders with specified gap (0 = touching).

    Returns (points, labels, normals) where:
    - labels == 0: cylinder 1
    - labels == 1: cylinder 2
    """
    if rng is None:
        rng = np.random.default_rng(42)

    theta = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
    z = np.linspace(0, 0.5, n_rings)

    pts1, n1, pts2, n2 = [], [], [], []
    separation = 2 * radius + gap

    for zi in z:
        for t in theta:
            # Cylinder 1 centered at x=0
            x1 = radius * np.cos(t)
            y1 = radius * np.sin(t)
            pts1.append([x1, y1, zi])
            n1.append([np.cos(t), np.sin(t), 0.0])

            # Cylinder 2 centered at x=separation
            x2 = separation + radius * np.cos(t)
            y2 = radius * np.sin(t)
            pts2.append([x2, y2, zi])
            n2.append([np.cos(t), np.sin(t), 0.0])

    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)
    n1 = np.array(n1, dtype=float)
    n2 = np.array(n2, dtype=float)

    if noise_std > 0:
        pts1 += rng.normal(0, noise_std, pts1.shape)
        pts2 += rng.normal(0, noise_std, pts2.shape)
        # Noisy normals — renormalize
        n1 += rng.normal(0, noise_std * 2, n1.shape)
        n2 += rng.normal(0, noise_std * 2, n2.shape)
        n1 /= np.linalg.norm(n1, axis=1, keepdims=True).clip(1e-8)
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True).clip(1e-8)

    pts = np.vstack([pts1, pts2])
    normals = np.vstack([n1, n2])
    labels_gt = np.array([0] * len(pts1) + [1] * len(pts2))
    return pts, labels_gt, normals


def generate_single_pipe_with_weld_seam(
    n_per_ring: int = 24,
    n_rings: int = 25,
    radius: float = 0.05,
    weld_normal_jitter_deg: float = 25.0,
    noise_std: float = 0.002,
    rng: np.random.Generator = None,
) -> tuple:
    """Generate a single pipe with a weld-seam ring that has exaggerated normal variation.

    Tests whether too-tight normal threshold causes false splitting of a single instance.
    Returns (points, labels_gt, normals) where all labels_gt == 0 (one instance).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    theta = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
    z = np.linspace(0, 0.5, n_rings)
    weld_z = z[n_rings // 2]  # Weld seam at the midpoint

    pts, normals_list = [], []
    for zi in z:
        for t in theta:
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            pts.append([x, y, zi])
            nx, ny = np.cos(t), np.sin(t)
            if abs(zi - weld_z) < 0.015:  # Weld zone: exaggerated normal noise
                jitter = np.radians(weld_normal_jitter_deg)
                nx += rng.uniform(-jitter, jitter)
                ny += rng.uniform(-jitter, jitter)
                norm = np.sqrt(nx ** 2 + ny ** 2)
                nx, ny = nx / max(norm, 1e-8), ny / max(norm, 1e-8)
            normals_list.append([nx, ny, 0.0])

    pts = np.array(pts, dtype=float)
    normals_arr = np.array(normals_list, dtype=float)
    if noise_std > 0:
        pts += rng.normal(0, noise_std, pts.shape)
    labels_gt = np.zeros(len(pts), dtype=int)
    return pts, labels_gt, normals_arr


def evaluate_clustering(labels_pred: np.ndarray, labels_gt: np.ndarray, n_gt_instances: int):
    """Compute instance count accuracy, false-merge rate, and false-split rate."""
    n_pred = len(set(labels_pred[labels_pred >= 0]))

    if n_gt_instances == 1:
        # Single-instance test: any split is a false split
        false_split_rate = max(0, n_pred - 1) / n_gt_instances
        false_merge_rate = 0.0
        correct = (n_pred == 1)
    elif n_gt_instances == 2:
        # Two-instance test: merged into 1 is a false merge, fragmented is false split
        false_merge_rate = 1.0 if n_pred < n_gt_instances else 0.0
        false_split_rate = max(0, n_pred - n_gt_instances) / n_gt_instances
        correct = (n_pred == n_gt_instances)
    else:
        false_merge_rate = max(0, n_gt_instances - n_pred) / n_gt_instances
        false_split_rate = max(0, n_pred - n_gt_instances) / n_gt_instances
        correct = (n_pred == n_gt_instances)

    return {
        "n_pred_instances": n_pred,
        "n_gt_instances": n_gt_instances,
        "correct": correct,
        "false_merge_rate": false_merge_rate,
        "false_split_rate": false_split_rate,
    }


def main():
    print("=" * 70)
    print("NORMAL-CONSISTENCY DBSCAN PARAMETER SWEEP")
    print("=" * 70)

    normal_thresholds = [30, 40, 50, 60, 70]  # degrees
    eps_values = [0.01, 0.02, 0.05, 0.10]
    noise_stds = [0.0, 0.002]  # 0mm, 2mm noise

    rng = np.random.default_rng(42)
    results = []

    # --- Scenario A: Two touching cylinders (should be separated) ---
    print("\n[Scenario A] Two touching cylinders (false-merge test)")
    for noise_std in noise_stds:
        pts, labels_gt, normals = generate_touching_cylinders(noise_std=noise_std, rng=rng)
        for eps in eps_values:
            for normal_thresh in normal_thresholds:
                # Euclidean baseline
                labels_eucl = dbscan(pts, eps=eps, min_samples=3)
                # Normal-consistent
                labels_norm = dbscan(pts, eps=eps, min_samples=3,
                                     normals=normals, max_normal_angle_deg=float(normal_thresh))
                eval_eucl = evaluate_clustering(labels_eucl, labels_gt, n_gt_instances=2)
                eval_norm = evaluate_clustering(labels_norm, labels_gt, n_gt_instances=2)
                results.append({
                    "scenario": "touching_cylinders",
                    "noise_mm": noise_std * 1000,
                    "eps": eps,
                    "normal_threshold_deg": normal_thresh,
                    "euclidean_dbscan": eval_eucl,
                    "normal_consistent_dbscan": eval_norm,
                })

    # --- Scenario B: Single pipe with weld-seam normal jitter (false-split test) ---
    print("\n[Scenario B] Single pipe with weld-seam normal variation (false-split test)")
    for weld_jitter in [15.0, 25.0, 40.0]:
        pts, labels_gt, normals = generate_single_pipe_with_weld_seam(
            weld_normal_jitter_deg=weld_jitter, rng=rng
        )
        for eps in [0.02, 0.05]:
            for normal_thresh in normal_thresholds:
                labels_norm = dbscan(pts, eps=eps, min_samples=3,
                                     normals=normals, max_normal_angle_deg=float(normal_thresh))
                eval_norm = evaluate_clustering(labels_norm, labels_gt, n_gt_instances=1)
                results.append({
                    "scenario": "single_pipe_weld_seam",
                    "weld_jitter_deg": weld_jitter,
                    "eps": eps,
                    "normal_threshold_deg": normal_thresh,
                    "euclidean_dbscan": None,
                    "normal_consistent_dbscan": eval_norm,
                })

    # Save JSON
    out_dir = Path("research_audit")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "clustering_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Summary Table
    print("\n[Scenario A Summary] Two touching cylinders:")
    print(f"{'Noise':<10} {'eps':<8} {'NormalThresh':<15} {'Eucl Correct':<14} {'Norm Correct':<14} {'FalseMerge':<12} {'FalseSplit':<12}")
    print("-" * 90)
    for r in results:
        if r["scenario"] != "touching_cylinders":
            continue
        en = r["euclidean_dbscan"]
        nn = r["normal_consistent_dbscan"]
        print(
            f"{r['noise_mm']:.1f} mm   {r['eps']:<8.3f} {r['normal_threshold_deg']:>5}°          "
            f"{'Y' if en['correct'] else 'N':<14} {'Y' if nn['correct'] else 'N':<14} "
            f"{nn['false_merge_rate']:<12.2f} {nn['false_split_rate']:<12.2f}"
        )

    print("\n[Scenario B Summary] Single pipe with weld-seam (false-split test):")
    print(f"{'Weld Jitter':<14} {'eps':<8} {'NormalThresh':<15} {'Correct (no split)':<20} {'FalseSplit':<12}")
    print("-" * 75)
    for r in results:
        if r["scenario"] != "single_pipe_weld_seam":
            continue
        nn = r["normal_consistent_dbscan"]
        print(
            f"{r['weld_jitter_deg']:>5}°         {r['eps']:<8.3f} {r['normal_threshold_deg']:>5}°          "
            f"{'Y' if nn['correct'] else 'N':<20} {nn['false_split_rate']:<12.2f}"
        )

    # Write Markdown table
    md = "# Normal-Consistency DBSCAN Parameter Sweep\n\n"
    md += "Systematic sweep validating the normal-angle threshold (50°) used in normal-consistent DBSCAN.\n"
    md += "Tests two scenarios: (A) false-merge prevention on touching cylinders, (B) false-split on single pipe with weld-seam normal jitter.\n\n"
    md += "## Scenario A: Touching Cylinders (False-Merge Test)\n\n"
    md += "| Noise | eps | Normal Thresh | Euclidean Correct | Normal-Consistent Correct | False-Merge | False-Split |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for r in results:
        if r["scenario"] != "touching_cylinders":
            continue
        en = r["euclidean_dbscan"]
        nn = r["normal_consistent_dbscan"]
        md += (
            f"| {r['noise_mm']:.1f} mm | {r['eps']:.3f} | {r['normal_threshold_deg']}° | "
            f"{'✓' if en['correct'] else '✗'} | **{'✓' if nn['correct'] else '✗'}** | "
            f"{nn['false_merge_rate']:.2f} | {nn['false_split_rate']:.2f} |\n"
        )
    md += "\n## Scenario B: Single Pipe with Weld-Seam Normal Jitter (False-Split Test)\n\n"
    md += "| Weld Jitter | eps | Normal Thresh | No False Split | False-Split Rate |\n"
    md += "|---|---|---|---|---|\n"
    for r in results:
        if r["scenario"] != "single_pipe_weld_seam":
            continue
        nn = r["normal_consistent_dbscan"]
        md += (
            f"| {r['weld_jitter_deg']}° | {r['eps']:.3f} | {r['normal_threshold_deg']}° | "
            f"**{'✓' if nn['correct'] else '✗'}** | {nn['false_split_rate']:.2f} |\n"
        )

    with open(out_dir / "clustering_sweep.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nClustering sweep results saved to research_audit/clustering_sweep.json and clustering_sweep.md")


if __name__ == "__main__":
    main()
