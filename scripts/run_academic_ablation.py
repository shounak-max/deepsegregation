"""Run rigorous ablation experiments comparing baselines on noisy and occluded data."""

import json
import time
from pathlib import Path
import numpy as np

from deepsegregation.baselines import compare_elbow_fitting_methods, run_ransac_only_pipeline
from deepsegregation.cylinder import CylinderFit
from deepsegregation.pointcloud import PointCloud
from deepsegregation.pipeline import run_pipeline


def generate_assembly(major_R=0.25, minor_r=0.04, noise_std=0.0, occlusion_pct=0.0, random_state=42):
    rng = np.random.default_rng(random_state)

    # 1. Straight cylinder 1 along +Y at x = major_R
    y1 = np.linspace(-0.3, 0.0, 20)
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts1 = []
    for y in y1:
        for t in theta:
            x = major_R + minor_r * np.cos(t)
            z = minor_r * np.sin(t)
            pts1.append([x, y, z])
    pts1 = np.array(pts1)

    # 2. Elbow: 90-degree bend connecting (major_R, 0, 0) to (0, major_R, 0)
    max_phi = (np.pi / 2) * (1.0 - occlusion_pct)
    phi = np.linspace(0, max_phi, max(6, int(20 * (1.0 - occlusion_pct))))
    pts_elbow = []
    for p in phi:
        for t in theta:
            x = (major_R + minor_r * np.cos(t)) * np.cos(p)
            y = (major_R + minor_r * np.cos(t)) * np.sin(p)
            z = minor_r * np.sin(t)
            pts_elbow.append([x, y, z])
    pts_elbow = np.array(pts_elbow)

    # 3. Straight cylinder 2 along -X at y = major_R
    x2 = np.linspace(-0.3, 0.0, 20)
    pts2 = []
    for x in x2:
        for t in theta:
            y = major_R + minor_r * np.cos(t)
            z = minor_r * np.sin(t)
            pts2.append([x, y, z])
    pts2 = np.array(pts2)

    all_pts = np.vstack([pts1, pts_elbow, pts2])
    labels = np.concatenate([
        np.full(len(pts1), 1),
        np.full(len(pts_elbow), 2),
        np.full(len(pts2), 1),
    ])

    if noise_std > 0:
        all_pts += rng.normal(0, noise_std, size=all_pts.shape)
        pts_elbow += rng.normal(0, noise_std, size=pts_elbow.shape)

    cyl_a = CylinderFit(
        center=np.array([major_R, -0.15, 0.0]),
        axis=np.array([0.0, 1.0, 0.0]),
        radius=minor_r,
        inlier_mask=np.ones(len(pts1), dtype=bool),
        residuals=np.zeros(len(pts1)),
        iterations=1,
    )
    cyl_b = CylinderFit(
        center=np.array([-0.15, major_R, 0.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        radius=minor_r,
        inlier_mask=np.ones(len(pts2), dtype=bool),
        residuals=np.zeros(len(pts2)),
        iterations=1,
    )

    return all_pts, labels, pts_elbow, cyl_a, cyl_b, major_R, minor_r


def main():
    print("=" * 70)
    print("RUNNING ACADEMIC BASELINE & NOISE ABLATION MATRIX")
    print("=" * 70)

    noise_levels = [0.0, 0.002, 0.005]  # 0mm, 2mm, 5mm (typical LiDAR noise)
    occlusion_levels = [0.0, 0.30]       # Full view, 30% occluded arc

    results = []

    for noise in noise_levels:
        for occ in occlusion_levels:
            print(f"\n--- Testing Noise = {noise*1000:.1f}mm | Occlusion = {occ*100:.0f}% ---")
            pts, labels, elbow_pts, cyl_a, cyl_b, gt_R, gt_r = generate_assembly(
                noise_std=noise, occlusion_pct=occ, random_state=42
            )

            # Compare elbow fitting methods (three-way information-equal ablation)
            res_u, res_p, res_t = compare_elbow_fitting_methods(
                elbow_pts, cyl_a, cyl_b,
                ground_truth_major_r=gt_R,
                ground_truth_minor_r=gt_r,
                distance_threshold=max(0.01, 2.5 * noise),
            )

            # Compare RANSAC-only on the whole cloud
            cloud = PointCloud(pts)
            t0 = time.perf_counter()
            ransac_fits = run_ransac_only_pipeline(cloud, distance_threshold=max(0.01, 2.5 * noise))
            t_ransac = time.perf_counter() - t0

            # Compare End-to-End Pipeline with Topological Fitting
            t0 = time.perf_counter()
            pipe_res = run_pipeline(cloud, labels, fit_threshold=max(0.01, 2.5 * noise))
            t_pipeline = time.perf_counter() - t0

            entry = {
                "noise_mm": noise * 1000,
                "occlusion_pct": occ * 100,
                "unconstrained_torus": {
                    "major_r_err_pct": res_u.bend_radius_error_pct,
                    "minor_r_err_pct": res_u.mean_radius_error_pct,
                    "runtime_sec": res_u.runtime_sec,
                    "success": res_u.success,
                    "inlier_rmse": res_u.inlier_rmse,
                },
                "prior_initialized_torus": {
                    "major_r_err_pct": res_p.bend_radius_error_pct,
                    "minor_r_err_pct": res_p.mean_radius_error_pct,
                    "runtime_sec": res_p.runtime_sec,
                    "success": res_p.success,
                    "inlier_rmse": res_p.inlier_rmse,
                },
                "topological_torus": {
                    "major_r_err_pct": res_t.bend_radius_error_pct,
                    "minor_r_err_pct": res_t.mean_radius_error_pct,
                    "runtime_sec": res_t.runtime_sec,
                    "success": res_t.success,
                    "inlier_rmse": res_t.inlier_rmse,
                },
                "ransac_only_baseline": {
                    "detected_cylinders": len(ransac_fits),
                    "runtime_sec": t_ransac,
                },
                "hybrid_topological_pipeline": {
                    "detected_instances": len(pipe_res.instances),
                    "fitted_primitives": len(pipe_res.fits),
                    "runtime_sec": t_pipeline,
                }
            }
            results.append(entry)

            print(f"  Unconstrained Torus RANSAC  : Bend Err = {res_u.bend_radius_error_pct:.2f}% | RMSE = {res_u.inlier_rmse:.4f}m | Time = {res_u.runtime_sec:.3f}s")
            print(f"  Prior-Initialized RANSAC    : Bend Err = {res_p.bend_radius_error_pct:.2f}% | RMSE = {res_p.inlier_rmse:.4f}m | Time = {res_p.runtime_sec:.3f}s")
            print(f"  Topological Graph Torus     : Bend Err = {res_t.bend_radius_error_pct:.2f}% | RMSE = {res_t.inlier_rmse:.4f}m | Time = {res_t.runtime_sec:.3f}s")
            print(f"  RANSAC-only detected {len(ransac_fits)} primitives in {t_ransac:.3f}s vs Hybrid Pipeline {len(pipe_res.fits)} primitives in {t_pipeline:.3f}s")

    # Save results
    out_dir = Path("research_audit")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Table with all three methods + honest RMSE
    md = "# Empirical Baseline & Ablation Study Results\n\n"
    md += "Three-way information-equal ablation: Unconstrained vs. Prior-Initialized (same axis info, no hard constraint) vs. Topological (same axis info + C1 constraint).\n\n"
    md += "| Noise (mm) | Occlusion | Unconstrained Bend Err | Prior-Init Bend Err | Topological Bend Err | Unconstrained RMSE | Prior-Init RMSE | Topological RMSE | Speedup (vs Unconstrained) |\n"
    md += "|---|---|---|---|---|---|---|---|---|\n"
    for r in results:
        u_err = f"{r['unconstrained_torus']['major_r_err_pct']:.2f}%" if r['unconstrained_torus']['major_r_err_pct'] is not None else "FAIL"
        p_err = f"{r['prior_initialized_torus']['major_r_err_pct']:.2f}%" if r['prior_initialized_torus']['major_r_err_pct'] is not None else "FAIL"
        t_err = f"{r['topological_torus']['major_r_err_pct']:.2f}%"
        u_rmse = f"{r['unconstrained_torus']['inlier_rmse'] * 1000:.3f} mm" if r['unconstrained_torus']['inlier_rmse'] < 1e9 else "N/A"
        p_rmse = f"{r['prior_initialized_torus']['inlier_rmse'] * 1000:.3f} mm" if r['prior_initialized_torus']['inlier_rmse'] < 1e9 else "N/A"
        t_rmse = f"{r['topological_torus']['inlier_rmse'] * 1000:.3f} mm"
        speedup = f"{r['unconstrained_torus']['runtime_sec'] / max(1e-4, r['topological_torus']['runtime_sec']):.1f}x"
        md += f"| {r['noise_mm']:.1f} mm | {r['occlusion_pct']:.0f}% | {u_err} | {p_err} | **{t_err}** | {u_rmse} | {p_rmse} | **{t_rmse}** | **{speedup}** |\n"

    with open(out_dir / "ablation_results.md", "w") as f:
        f.write(md)

    print("\nAblation results successfully saved to research_audit/ablation_results.json and research_audit/ablation_results.md")


if __name__ == "__main__":
    main()
