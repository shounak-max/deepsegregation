"""Benchmark baselines for pipe segmentation and primitive fitting.

Provides:
1. RANSAC-only baseline (no deep learning semantic segmentation).
2. Unconstrained Torus RANSAC vs. Topological Graph Torus Fitting ablation.
3. Information-equal ablation: unconstrained RANSAC with cylinder-axis priors.
4. Quantitative evaluation harness under noise and partial occlusion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cylinder import CylinderFit, fit_cylinder_ransac
from .pointcloud import PointCloud
from .topology import ElbowGeometryError, TopologicalElbowFit, fit_torus_topological
from .torus import TorusFit, fit_torus_ransac, torus_residuals


@dataclass(frozen=True)
class BaselineResult:
    method_name: str
    runtime_sec: float
    detected_primitives: int
    mean_radius_error_pct: float
    bend_radius_error_pct: Optional[float]
    # Honest empirical RMSE: how well the fitted surface explains the point cloud.
    # This is distinct from C1 tangent error (which is a constraint property,
    # not a measurement of fit quality vs raw data).
    inlier_rmse: float
    inlier_ratio: float
    success: bool


def run_ransac_only_pipeline(
    cloud: PointCloud,
    *,
    distance_threshold: float = 0.01,
    max_primitives: int = 10,
    min_points: int = 50,
) -> List[CylinderFit]:
    """Pure classical RANSAC baseline: iteratively extract cylinders without DL."""
    remaining_points = cloud.points.copy()
    fits: List[CylinderFit] = []

    for _ in range(max_primitives):
        if len(remaining_points) < min_points:
            break
        try:
            fit = fit_cylinder_ransac(
                remaining_points,
                distance_threshold=distance_threshold,
                max_trials=300,
                min_inliers=min_points,
            )
        except RuntimeError:
            break

        fits.append(fit)
        # Remove inliers
        inlier_indices = np.flatnonzero(fit.inlier_mask)
        if len(inlier_indices) == 0:
            break
        remaining_mask = ~fit.inlier_mask
        remaining_points = remaining_points[remaining_mask]

    return fits


def _compute_inlier_rmse(
    pts: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    major_r: float,
    minor_r: float,
    distance_threshold: float,
) -> float:
    """Compute point-cloud RMSE for inlier points only."""
    res = torus_residuals(pts, center, axis, major_r, minor_r)
    inlier_res = res[res <= distance_threshold]
    return float(np.sqrt(np.mean(inlier_res**2))) if len(inlier_res) else float("inf")


def compare_elbow_fitting_methods(
    elbow_points: np.ndarray,
    connecting_cylinder_a: CylinderFit,
    connecting_cylinder_b: CylinderFit,
    ground_truth_major_r: float,
    ground_truth_minor_r: float,
    *,
    distance_threshold: float = 0.01,
) -> Tuple[BaselineResult, BaselineResult, BaselineResult]:
    """Three-way ablation: Unconstrained vs. Prior-Initialized vs. Topological.

    Returns (unconstrained_result, prior_initialized_result, topological_result).

    **Why three methods?**
    The key scientific question is whether the *topological constraint* adds value
    beyond simply providing a good axis initialization. The information-equal
    comparison (prior_initialized) answers this:
    - Unconstrained: no axis prior, no constraint.
    - Prior-initialized: axis prior from cylinders, but still 7-DoF free optimization.
    - Topological: axis prior + hard C1 constraint + bounded 1D/2D manifold search.

    If prior_initialized ≈ topological, the contribution is initialization quality.
    If topological >> prior_initialized, the constraint itself drives the improvement.
    """
    pts = np.asarray(elbow_points, dtype=float)

    # --- Method 1: Unconstrained Torus RANSAC (no prior information) ---
    t0 = time.perf_counter()
    unconstrained_success = False
    unconstrained_major_err = 100.0
    unconstrained_minor_err = 100.0
    inlier_ratio_unconstrained = 0.0
    inlier_rmse_unconstrained = float("inf")

    try:
        u_fit = fit_torus_ransac(
            pts,
            distance_threshold=distance_threshold,
            max_trials=1000,
            random_state=42,
        )
        unconstrained_success = True
        unconstrained_major_err = abs(u_fit.major_radius - ground_truth_major_r) / ground_truth_major_r * 100.0
        unconstrained_minor_err = abs(u_fit.minor_radius - ground_truth_minor_r) / ground_truth_minor_r * 100.0
        inlier_ratio_unconstrained = float(u_fit.inlier_count / len(pts))
        inlier_rmse_unconstrained = _compute_inlier_rmse(
            pts, u_fit.center, u_fit.axis, u_fit.major_radius, u_fit.minor_radius, distance_threshold
        )
    except Exception:
        pass

    t1 = time.perf_counter()
    res_unconstrained = BaselineResult(
        method_name="Unconstrained_Torus_RANSAC",
        runtime_sec=t1 - t0,
        detected_primitives=1 if unconstrained_success else 0,
        mean_radius_error_pct=unconstrained_minor_err,
        bend_radius_error_pct=unconstrained_major_err,
        inlier_rmse=inlier_rmse_unconstrained,
        inlier_ratio=inlier_ratio_unconstrained,
        success=unconstrained_success,
    )

    # --- Method 2: Prior-Initialized Torus RANSAC (same cylinder-axis information,
    #     but unconstrained 7-DoF optimization) ---
    # This is the information-equal comparison.  If the topological method wins
    # over THIS baseline, the constraint itself (not just axis initialization) is
    # responsible for the improvement.
    t2 = time.perf_counter()
    prior_success = False
    prior_major_err = 100.0
    prior_minor_err = 100.0
    inlier_ratio_prior = 0.0
    inlier_rmse_prior = float("inf")

    try:
        # Derive the ideal axis prior from the two cylinder axes (same information
        # that the topological method uses)
        ax_a = connecting_cylinder_a.axis / np.linalg.norm(connecting_cylinder_a.axis)
        ax_b = connecting_cylinder_b.axis / np.linalg.norm(connecting_cylinder_b.axis)
        ideal_axis_prior = np.cross(ax_a, ax_b)
        prior_norm = np.linalg.norm(ideal_axis_prior)
        if prior_norm > 1e-6:
            ideal_axis_prior = ideal_axis_prior / prior_norm
        else:
            ideal_axis_prior = None  # Degenerate — fall back to pure unconstrained

        p_fit = fit_torus_ransac(
            pts,
            axis=ideal_axis_prior,  # Axis is fixed to cylinder cross-product (soft prior)
            distance_threshold=distance_threshold,
            max_trials=1000,
            random_state=42,
        )
        prior_success = True
        prior_major_err = abs(p_fit.major_radius - ground_truth_major_r) / ground_truth_major_r * 100.0
        prior_minor_err = abs(p_fit.minor_radius - ground_truth_minor_r) / ground_truth_minor_r * 100.0
        inlier_ratio_prior = float(p_fit.inlier_count / len(pts))
        inlier_rmse_prior = _compute_inlier_rmse(
            pts, p_fit.center, p_fit.axis, p_fit.major_radius, p_fit.minor_radius, distance_threshold
        )
    except Exception:
        pass

    t3 = time.perf_counter()
    res_prior = BaselineResult(
        method_name="Prior_Initialized_Torus_RANSAC",
        runtime_sec=t3 - t2,
        detected_primitives=1 if prior_success else 0,
        mean_radius_error_pct=prior_minor_err,
        bend_radius_error_pct=prior_major_err,
        inlier_rmse=inlier_rmse_prior,
        inlier_ratio=inlier_ratio_prior,
        success=prior_success,
    )

    # --- Method 3: Topological Graph-Constrained Torus (hard constraint) ---
    t4 = time.perf_counter()
    topological_success = False
    topological_major_err = 100.0
    topological_minor_err = 100.0
    inlier_ratio_topological = 0.0
    inlier_rmse_topological = float("inf")

    try:
        t_fit = fit_torus_topological(
            pts,
            connecting_cylinder_a,
            connecting_cylinder_b,
            distance_threshold=distance_threshold,
        )
        topological_success = True
        topological_major_err = abs(t_fit.major_radius - ground_truth_major_r) / ground_truth_major_r * 100.0
        topological_minor_err = abs(t_fit.minor_radius - ground_truth_minor_r) / ground_truth_minor_r * 100.0
        inlier_ratio_topological = float(t_fit.inlier_count / len(pts))
        inlier_rmse_topological = t_fit.inlier_rmse
    except ElbowGeometryError:
        # Topological method cannot handle this case (shallow angle, etc.)
        topological_success = False
    except Exception:
        pass

    t5 = time.perf_counter()
    res_topological = BaselineResult(
        method_name="Topological_Graph_Torus",
        runtime_sec=t5 - t4,
        detected_primitives=1 if topological_success else 0,
        mean_radius_error_pct=topological_minor_err,
        bend_radius_error_pct=topological_major_err,
        inlier_rmse=inlier_rmse_topological,
        inlier_ratio=inlier_ratio_topological,
        success=topological_success,
    )

    return res_unconstrained, res_prior, res_topological
