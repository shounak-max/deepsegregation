"""Robust torus fitting for segmented pipe elbows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TorusFit:
    """Estimated torus parameters and diagnostics."""

    center: Array
    axis: Array
    major_radius: float
    minor_radius: float
    inlier_mask: Array
    residuals: Array
    iterations: int  # Total trial budget evaluated
    best_trial: int = 0  # 1-based index of the best RANSAC trial

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def rmse(self) -> float:
        values = self.residuals[self.inlier_mask]
        return float(np.sqrt(np.mean(values**2))) if values.size else float("inf")


def _as_points(points: Array) -> Array:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if values.shape[0] < 6:
        raise ValueError("at least 6 points are required to fit a torus")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    return values


def _unit(vector: Array) -> Array:
    result = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(result))
    if length < 1e-12:
        raise ValueError("axis must be non-zero")
    result = result / length
    first = int(np.argmax(np.abs(result)))
    return -result if result[first] < 0 else result


def _plane_basis(axis: Array) -> Tuple[Array, Array]:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, axis))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = helper - axis * np.dot(helper, axis)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _pca_axis(points: Array) -> Array:
    centered = points - np.median(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return _unit(vh[-1])


def _circle_from_three(sample: Array) -> Optional[Tuple[float, float, float]]:
    x1, y1 = sample[0]
    x2, y2 = sample[1]
    x3, y3 = sample[2]
    matrix = np.array([[2 * (x2 - x1), 2 * (y2 - y1)], [2 * (x3 - x1), 2 * (y3 - y1)]])
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        return None
    rhs = np.array([x2*x2 + y2*y2 - x1*x1 - y1*y1,
                    x3*x3 + y3*y3 - x1*x1 - y1*y1])
    center = np.linalg.solve(matrix, rhs)
    radius = float(np.linalg.norm(center - sample[0]))
    return (float(center[0]), float(center[1]), radius) if radius > 1e-9 else None


def _project(points: Array, origin: Array, axis: Array) -> Tuple[Array, Array, Array, Array]:
    u, v = _plane_basis(axis)
    relative = points - origin
    return np.column_stack((relative @ u, relative @ v)), relative @ axis, u, v


def torus_residuals(points: Array, center: Array, axis: Array,
                    major_radius: float, minor_radius: float) -> Array:
    """Return absolute point-to-torus surface distances."""
    values = _as_points(points)
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center must have shape (3,)")
    if major_radius <= 0 or minor_radius <= 0:
        raise ValueError("major_radius and minor_radius must be positive")
    axis = _unit(axis)
    relative = values - center
    axial = relative @ axis
    radial = relative - np.outer(axial, axis)
    rho = np.linalg.norm(radial, axis=1)
    return np.abs(np.sqrt((rho - major_radius)**2 + axial**2) - minor_radius)


def _fit_for_axis(points: Array, axis: Array, distance_threshold: float,
                  max_trials: int, min_inliers: int,
                  rng: np.random.Generator) -> Optional[TorusFit]:
    origin = np.median(points, axis=0)
    projected, z, u, v = _project(points, origin, axis)
    extent = float(np.max(np.linalg.norm(projected, axis=1)))
    if extent < 1e-8:
        return None

    best = None
    for trial in range(max_trials):
        circle = _circle_from_three(projected[rng.choice(len(points), 3, replace=False)])
        if circle is None:
            continue
        cx, cy, major = circle
        if major > 10 * extent:
            continue
        rho = np.sqrt((projected[:, 0] - cx)**2 + (projected[:, 1] - cy)**2)
        minor = float(np.median(np.sqrt((rho - major)**2 + z**2)))
        if minor <= 1e-9 or minor > 2 * extent or major <= minor:
            continue
        residuals = np.abs(np.sqrt((rho - major)**2 + z**2) - minor)
        inliers = residuals <= distance_threshold
        score = (int(np.count_nonzero(inliers)),
                 -float(np.median(residuals[inliers])) if inliers.any() else -float("inf"))
        if best is None or score > best[0]:
            best = (score, cx, cy, major, minor, trial + 1)

    initial_parameters = []
    best_trials = 0
    if best is not None:
        _, cx, cy, major, minor, best_trials = best
        initial_parameters.append((cx, cy, major, minor))
    _, _, vh = np.linalg.svd(projected - np.mean(projected, axis=0), full_matrices=False)
    curvature_direction = vh[-1]
    center_seed = np.mean(projected, axis=0)
    for scale in (0.5, 1.0, 2.0, 4.0):
        offset = scale * extent * curvature_direction
        for sign in (-1.0, 1.0):
            initial_parameters.append((
                float(center_seed[0] + sign * offset[0]),
                float(center_seed[1] + sign * offset[1]),
                max(extent * scale, distance_threshold * 2),
                max(extent * 0.1, distance_threshold * 2),
            ))

    try:
        from scipy.optimize import least_squares
    except ImportError:
        if best is None:
            return None
        _, cx, cy, major, minor, trials = best
        rho = np.sqrt((projected[:, 0] - cx)**2 + (projected[:, 1] - cy)**2)
        residuals = np.abs(np.sqrt((rho - major)**2 + z**2) - minor)
        inliers = residuals <= distance_threshold
        if int(np.count_nonzero(inliers)) < min_inliers:
            return None
        return TorusFit(origin + cx*u + cy*v, axis, major, minor, inliers, residuals, max_trials, best_trial=trials)

    lower = np.array([-20*extent, -20*extent, 1e-8, 1e-8])
    upper = np.array([20*extent, 20*extent, 20*extent, 2*extent])
    refined = []
    for initial in initial_parameters:
        initial = np.clip(np.asarray(initial, dtype=float), lower + 1e-8, upper - 1e-8)

        def residual_function(parameters):
            cx, cy, major, minor = parameters
            rho = np.sqrt((projected[:, 0] - cx)**2 + (projected[:, 1] - cy)**2)
            return np.sqrt((rho - major)**2 + z**2) - minor

        try:
            solution = least_squares(residual_function, initial, bounds=(lower, upper),
                                     loss="cauchy", f_scale=distance_threshold, max_nfev=300)
        except (ValueError, np.linalg.LinAlgError):
            continue
        cx, cy, major, minor = solution.x
        if major <= minor:
            continue
        residuals = np.abs(residual_function(solution.x))
        inliers = residuals <= distance_threshold
        if int(np.count_nonzero(inliers)) >= min_inliers:
            refined.append(TorusFit(origin + cx*u + cy*v, axis, float(major), float(minor),
                                     inliers, residuals, max_trials, best_trial=best_trials))
    if not refined:
        return None

    # Correct a small PCA-axis error using the consensus model itself.
    lower_full = np.concatenate((np.min(points, axis=0) - extent, np.full(3, -1.), [1e-8, 1e-8]))
    upper_full = np.concatenate((np.max(points, axis=0) + extent, np.full(3, 1.), [20*extent, 2*extent]))
    full_refined = []
    for fit in refined:
        initial = np.concatenate((fit.center, fit.axis, [fit.major_radius, fit.minor_radius]))

        def full_residual(parameters):
            center = parameters[:3]
            candidate_axis = parameters[3:6].copy()
            length = np.linalg.norm(candidate_axis)
            if length < 1e-8:
                return np.full(len(points), 1e3)
            candidate_axis /= length
            relative = points - center
            axial = relative @ candidate_axis
            radial = relative - np.outer(axial, candidate_axis)
            rho = np.linalg.norm(radial, axis=1)
            return np.sqrt((rho - parameters[6])**2 + axial**2) - parameters[7]

        try:
            solution = least_squares(full_residual, initial, bounds=(lower_full, upper_full),
                                     loss="cauchy", f_scale=distance_threshold, max_nfev=500)
        except (ValueError, np.linalg.LinAlgError):
            continue
        parameters = solution.x
        candidate_major, candidate_minor = float(parameters[6]), float(parameters[7])
        if candidate_major <= candidate_minor:
            continue
        candidate_axis = _unit(parameters[3:6])
        residuals = np.abs(full_residual(parameters))
        inliers = residuals <= distance_threshold
        if int(np.count_nonzero(inliers)) >= min_inliers:
            full_refined.append(TorusFit(parameters[:3], candidate_axis, candidate_major,
                                          candidate_minor, inliers, residuals, max_trials, best_trial=best_trials))
    return max(full_refined or refined, key=lambda fit: (fit.inlier_count, -fit.rmse))


def fit_torus_ransac(points: Array, *, distance_threshold: float = 0.01,
                     max_trials: int = 1200, min_inliers: Optional[int] = None,
                     min_inlier_ratio: float = 0.3,
                     axis: Optional[Array] = None,
                     random_state: Optional[int] = 0) -> TorusFit:
    """Fit a torus to an elbow point cloud using RANSAC and robust refinement."""
    values = _as_points(points)
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive")
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    if min_inliers is None:
        min_inliers = max(6, int(np.ceil(min_inlier_ratio * len(values))))
    if min_inliers < 6 or min_inliers > len(values):
        raise ValueError("min_inliers must be between 6 and the number of points")

    rng = np.random.default_rng(random_state)
    if axis is not None:
        axes = [_unit(axis)]
    else:
        # Pre-filter point cloud for robust initial axis estimation under outliers
        try:
            from .pointcloud import PointCloud
            from .preprocessing import remove_statistical_outliers
            cloud = PointCloud(values)
            k = min(16, max(2, len(values) - 1))
            filtered = remove_statistical_outliers(cloud, neighbors=k, z_threshold=1.5)
            clean_pts = filtered.points if len(filtered.points) >= 6 else values
        except Exception:
            clean_pts = values

        axes = [_pca_axis(clean_pts)]
        subset_size = max(8, int(0.7 * len(clean_pts)))
        for _ in range(5):
            axes.append(_pca_axis(clean_pts[rng.choice(len(clean_pts), subset_size, replace=False)]))

    trials_per_axis = max(200, max_trials // len(axes))
    candidates = []
    for candidate_axis in axes:
        candidate = _fit_for_axis(values, candidate_axis, distance_threshold,
                                  trials_per_axis, min_inliers, rng)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("RANSAC could not find a torus consensus set")
    return max(candidates, key=lambda fit: (fit.inlier_count, -fit.rmse))
