"""Demonstrate extraction of distinct industrial pipes from a point cloud scene

Using the GPU-trained baseline checkpoint to segment and separate:
1. Clutter/background rejection
2. Semantic labeling (straight pipe vs elbow)
3. DBSCAN instance separation of individual pipe instances
4. Precise RANSAC geometric parameter extraction (cylinder axis & radius, torus major/minor radii)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from deepsegregation.model import build_point_mlp, build_pointnet2_ssg
from deepsegregation.inference import predict_labels
from deepsegregation.pipeline import run_pipeline
from deepsegregation.synthetic import generate_scene, generate_elbow, generate_scene, PointCloud


def create_multi_pipe_scene(seed: int = 42):
    """Create a realistic industrial scene containing multiple straight pipes, an elbow, and background clutter."""
    rng = np.random.default_rng(seed)

    # 1. An elbow pipe (nominal minor_r=0.08m, major_R=0.65m)
    elbow_scan = generate_elbow(major_radius=0.65, minor_radius=0.08, bend_angle=np.pi/2, samples=1500, gaussian_noise=0.001, seed=seed)
    
    # 2. Straight pipe #1 (horizontal along X, nominal r=0.06m)
    x1 = rng.uniform(-0.8, 0.8, 800)
    phi1 = rng.uniform(0, 2 * np.pi, 800)
    p1 = np.column_stack((x1, 0.9 + 0.06 * np.cos(phi1), 0.06 * np.sin(phi1)))

    # 3. Straight pipe #2 (vertical along Z, nominal r=0.05m)
    z2 = rng.uniform(-0.6, 0.6, 600)
    phi2 = rng.uniform(0, 2 * np.pi, 600)
    p2 = np.column_stack((-0.8 + 0.05 * np.cos(phi2), -0.5 + 0.05 * np.sin(phi2), z2))

    # 4. Background clutter (valves/structural beams/noise)
    clutter = rng.uniform(-1.0, 1.0, (120, 3))

    all_points = np.vstack((elbow_scan.cloud.points, p1, p2, clutter))
    all_labels = np.concatenate((
        elbow_scan.cloud.labels,
        np.full(len(p1), 1, dtype=np.int64),
        np.full(len(p2), 1, dtype=np.int64),
        np.full(len(clutter), 0, dtype=np.int64),
    ))
    return PointCloud(all_points, labels=all_labels), {
        "elbow": {"major_R": 0.65, "minor_r": 0.08},
        "straight_1": {"r": 0.06},
        "straight_2": {"r": 0.05}
    }


def main():
    parser = argparse.ArgumentParser(description="Extract individual pipes from a collection of pipes")
    default_ckpt = "artifacts/point_mlp_3class/best.pt"
    if not Path(default_ckpt).exists():
        default_ckpt = "artifacts/point_mlp_k80/best.pt"
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint not found at {checkpoint_path}")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weights = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    is_pointnet2 = checkpoint.get("model_name") == "pointnet2_ssg" or any(k.startswith("sa1") for k in weights.keys())
    if is_pointnet2:
        model = build_pointnet2_ssg(input_features=3, num_classes=3)
    else:
        model = build_point_mlp(input_features=3, num_classes=3)

    try:
        model.load_state_dict(weights)
    except Exception as exc:
        print(f"Warning: strict checkpoint loading failed ({exc}), loading with strict=False")
        model.load_state_dict(weights, strict=False)
    model.eval()

    cloud, ground_truth = create_multi_pipe_scene(args.seed)
    total_points = len(cloud.points)
    print("=" * 75)
    print("PIPE EXTRACTION PIPELINE DEMONSTRATION")
    print("=" * 75)
    print(f"Total points in industrial scene: {total_points}")
    print(f"Ground truth items in scene:")
    print(f"  - Elbow:       Pipe radius = {ground_truth['elbow']['minor_r']:.3f}m, Bend radius = {ground_truth['elbow']['major_R']:.3f}m")
    print(f"  - Straight #1: Pipe radius = {ground_truth['straight_1']['r']:.3f}m")
    print(f"  - Straight #2: Pipe radius = {ground_truth['straight_2']['r']:.3f}m")
    print(f"  - Plus background clutter points")
    print("-" * 75)

    # Step 1: Semantic classification using GPU-trained checkpoint
    print(">> Step 1: Neural semantic classification (Straight / Elbow / Background)...")
    pred_labels = predict_labels(model, cloud.points, device="cpu")
    unique, counts = np.unique(pred_labels, return_counts=True)
    class_map = {0: "Background/Clutter", 1: "Straight Pipe", 2: "Elbow"}
    for u, c in zip(unique, counts):
        print(f"   * {class_map.get(u, str(u))}: {c} points ({c/total_points*100:.1f}%)")

    # Step 2: Instance separation (DBSCAN) & Step 3: RANSAC parameter fitting
    print("\n>> Step 2 & 3: Instance separation and RANSAC geometric extraction...")
    result = run_pipeline(cloud, semantic_labels=pred_labels, dbscan_eps=0.08, dbscan_min_samples=15, fit_threshold=0.015)

    print(f"   Successfully segmented into {len(result.instances)} distinct pipe instances!\n")
    for inst in result.instances:
        cls_name = "Elbow" if inst.semantic_class == 2 else "Straight Pipe"
        print(f"   [Instance ID {inst.instance_id}] Type: {cls_name}")
        print(f"     - Point cluster size: {len(inst.indices)} points")

        fit = result.fits.get(inst.instance_id)
        if fit is not None:
            if hasattr(fit, "minor_radius"):
                print(f"     - Torus Model Fit:")
                print(f"         Extracted Pipe Radius (r): {fit.minor_radius:.4f} m (Ground truth: {ground_truth['elbow']['minor_r']:.4f} m)")
                print(f"         Extracted Bend Radius (R): {fit.major_radius:.4f} m (Ground truth: {ground_truth['elbow']['major_R']:.4f} m)")
                print(f"         Inlier Points:             {fit.inlier_count} / {len(inst.indices)} ({fit.inlier_count/len(inst.indices)*100:.1f}%)")
                print(f"         Fit RMSE:                  {fit.rmse:.5f} m")
            elif hasattr(fit, "radius"):
                print(f"     - Cylinder Model Fit:")
                print(f"         Extracted Pipe Radius:     {fit.radius:.4f} m")
                print(f"         Axis Vector (direction):   [{fit.axis[0]:.2f}, {fit.axis[1]:.2f}, {fit.axis[2]:.2f}]")
                print(f"         Inlier Points:             {fit.inlier_count} / {len(inst.indices)} ({fit.inlier_count/len(inst.indices)*100:.1f}%)")
                print(f"         Fit RMSE:                  {fit.rmse:.5f} m")

    print("\n>> Step 4: Industrial Compliance Verification:")
    for rep in result.compliance:
        status_str = "PASS" if rep.passed is not False else "FAIL"
        print(f"   * Pipe #{rep.pipe_id} ({rep.geometry}): Estimated Radius = {rep.estimated_radius:.4f}m | Status: {status_str}")

    print("=" * 75)


if __name__ == "__main__":
    main()
