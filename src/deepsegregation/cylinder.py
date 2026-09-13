"""RANSAC cylinder fitting for straight pipe instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CylinderFit:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    inlier_mask: np.ndarray
    residuals: np.ndarray
    iterations: int

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def rmse(self) -> float:
        values = self.residuals[self.inlier_mask]
        return float(np.sqrt(np.mean(values**2))) if len(values) else float("inf")


def fit_cylinder_ransac(points: np.ndarray, *, distance_threshold: float = 0.005,
                        max_trials: int = 500, min_inliers: Optional[int] = None,
                        small_radius_threshold: float = 0.05,
                        random_state: Optional[int] = 0) -> CylinderFit:
    """Fit a cylinder; small-radius hypotheses automatically receive 3x trials."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 6:
        raise ValueError("points must have shape (N, 3) with N >= 6")
    if distance_threshold <= 0 or max_trials < 1:
        raise ValueError("distance_threshold must be positive and max_trials >= 1")
    if min_inliers is None:
        min_inliers = max(6, int(np.ceil(.5 * len(values))))
    centered = values - values.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    radial = values - np.outer(centered @ axis, axis) - values.mean(axis=0)
    # The provisional PCA cross-section gives a stable scale for the RANSAC
    # hypotheses and avoids choosing arbitrarily large radii.
    provisional_radius = float(np.median(np.linalg.norm(radial, axis=1)))
    trials = max_trials * (3 if provisional_radius < small_radius_threshold else 1)
    rng = np.random.default_rng(random_state)
    best = None
    origin = values.mean(axis=0)
    u = vh[1]
    v = vh[2]
    xy = np.column_stack(((values - origin) @ u, (values - origin) @ v))
    for trial in range(trials):
        sample = xy[rng.choice(len(values), 3, replace=False)]
        matrix = np.array([[2*(sample[1, 0]-sample[0, 0]), 2*(sample[1, 1]-sample[0, 1])],
                           [2*(sample[2, 0]-sample[0, 0]), 2*(sample[2, 1]-sample[0, 1])]])
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        rhs = np.array([np.dot(sample[1], sample[1])-np.dot(sample[0], sample[0]),
                        np.dot(sample[2], sample[2])-np.dot(sample[0], sample[0])])
        cx, cy = np.linalg.solve(matrix, rhs)
        radius = float(np.hypot(*(sample[0] - [cx, cy])))
        if not 0 < radius < 10 * max(provisional_radius, distance_threshold):
            continue
        distances = np.abs(np.hypot(xy[:, 0]-cx, xy[:, 1]-cy) - radius)
        mask = distances <= distance_threshold
        score = (int(mask.sum()), -float(np.median(distances[mask])) if mask.any() else -float("inf"))
        if best is None or score > best[0]:
            best = (score, cx, cy, radius, trial + 1)
    if best is None or best[0][0] < min_inliers:
        raise RuntimeError("RANSAC could not find a cylinder consensus set")
    _, cx, cy, radius, _ = best
    mask = np.zeros(len(values), dtype=bool)
    for _ in range(4):
        selected = xy[mask] if mask.any() else xy
        A = np.column_stack((2*selected[:, 0], 2*selected[:, 1], np.ones(len(selected))))
        solution, _, _, _ = np.linalg.lstsq(A, np.sum(selected**2, axis=1), rcond=None)
        cx, cy = solution[:2]
        radius = float(np.sqrt(max(1e-12, solution[2] + cx*cx + cy*cy)))
        distances = np.abs(np.hypot(xy[:, 0]-cx, xy[:, 1]-cy) - radius)
        mask = distances <= distance_threshold
    center = origin + cx*u + cy*v
    return CylinderFit(center, axis, radius, mask, distances, trials)
