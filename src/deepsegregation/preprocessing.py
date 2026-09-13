"""Preprocessing primitives used before segmentation and geometry fitting."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pointcloud import PointCloud


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    keys = np.floor(cloud.points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1 if len(inverse) else 0
    points = np.zeros((count, 3))
    np.add.at(points, inverse, cloud.points)
    sizes = np.bincount(inverse, minlength=count)
    points /= sizes[:, None]
    colors = None
    if cloud.colors is not None:
        colors = np.zeros((count, 3), dtype=np.float64)
        np.add.at(colors, inverse, cloud.colors)
        colors /= sizes[:, None]
        if np.issubdtype(cloud.colors.dtype, np.integer):
            colors = np.round(colors).astype(cloud.colors.dtype)
        else:
            colors = colors.astype(cloud.colors.dtype)
    normals = None
    if cloud.normals is not None:
        normals = np.zeros((count, 3))
        np.add.at(normals, inverse, cloud.normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-12)
    labels = None
    if cloud.labels is not None:
        labels = np.zeros(count, dtype=cloud.labels.dtype)
        for index in range(count):
            values, counts = np.unique(cloud.labels[inverse == index], return_counts=True)
            labels[index] = values[np.argmax(counts)]
    return PointCloud(points, colors=colors, normals=normals, labels=labels, metadata=dict(cloud.metadata))


def remove_statistical_outliers(cloud: PointCloud, neighbors: int = 16, z_threshold: float = 2.5) -> PointCloud:
    if neighbors < 2 or z_threshold <= 0:
        raise ValueError("neighbors must be >= 2 and z_threshold must be positive")
    from scipy.spatial import cKDTree
    distances, _ = cKDTree(cloud.points).query(cloud.points, k=min(neighbors + 1, len(cloud.points)))
    mean_distance = distances[:, 1:].mean(axis=1)
    cutoff = np.median(mean_distance) + z_threshold * np.std(mean_distance)
    return cloud.select(mean_distance <= cutoff)


def estimate_normals(cloud: PointCloud, neighbors: int = 24) -> PointCloud:
    if neighbors < 3:
        raise ValueError("neighbors must be >= 3")
    from scipy.spatial import cKDTree
    _, indices = cKDTree(cloud.points).query(cloud.points, k=min(neighbors, len(cloud.points)))
    normals = np.empty_like(cloud.points)
    for row, neighborhood in enumerate(cloud.points[indices]):
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        _, _, vh = np.linalg.svd(covariance, full_matrices=False)
        normal = vh[-1]
        # Make orientation deterministic relative to the scan centroid.
        if np.dot(normal, cloud.points[row] - cloud.points.mean(axis=0)) < 0:
            normal = -normal
        normals[row] = normal
    return PointCloud(cloud.points, cloud.colors, normals, cloud.labels, dict(cloud.metadata))


def preprocess(cloud: PointCloud, voxel_size: Optional[float] = None,
               denoise: bool = True, compute_normals: bool = True) -> PointCloud:
    result = cloud
    if voxel_size is not None:
        result = voxel_downsample(result, voxel_size)
    if denoise and len(result.points) >= 3:
        result = remove_statistical_outliers(result)
    if compute_normals and len(result.points) >= 3:
        result = estimate_normals(result)
    return result
