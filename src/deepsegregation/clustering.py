"""DBSCAN instance separation for semantic pipe/elbow points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .pointcloud import PointCloud


@dataclass(frozen=True)
class Instance:
    instance_id: int
    semantic_class: int
    indices: np.ndarray
    cloud: PointCloud


def dbscan(points: np.ndarray, eps: float = 0.05, min_samples: int = 12) -> np.ndarray:
    """Return DBSCAN labels; ``-1`` denotes noise."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if eps <= 0 or min_samples < 1:
        raise ValueError("eps must be positive and min_samples must be >= 1")
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, eps)
    labels = np.full(len(points), -1, dtype=np.int64)
    visited = np.zeros(len(points), dtype=bool)
    cluster_id = 0
    for seed in range(len(points)):
        if visited[seed]:
            continue
        visited[seed] = True
        if len(neighbors[seed]) < min_samples:
            continue
        labels[seed] = cluster_id
        queue = list(neighbors[seed])
        queued = set(queue)
        cursor = 0
        while cursor < len(queue):
            point = queue[cursor]
            cursor += 1
            if not visited[point]:
                visited[point] = True
                if len(neighbors[point]) >= min_samples:
                    for neighbor in neighbors[point]:
                        if neighbor not in queued:
                            queue.append(neighbor)
                            queued.add(neighbor)
            if labels[point] == -1:
                labels[point] = cluster_id
        cluster_id += 1
    return labels


def separate_instances(cloud: PointCloud, semantic_labels: np.ndarray,
                       eps: float = 0.05, min_samples: int = 12,
                       semantic_classes: Iterable[int] = (1, 2),
                       min_points: int = 30) -> List[Instance]:
    """Cluster each semantic class independently to avoid class merging."""
    labels = np.asarray(semantic_labels)
    if labels.shape != (len(cloud.points),):
        raise ValueError("semantic_labels must have one value per point")
    instances = []
    next_id = 0
    for semantic_class in semantic_classes:
        indices = np.flatnonzero(labels == semantic_class)
        if len(indices) == 0:
            continue
        cluster_labels = dbscan(cloud.points[indices], eps, min_samples)
        for cluster_id in sorted(set(cluster_labels.tolist())):
            if cluster_id < 0:
                continue
            member_indices = indices[cluster_labels == cluster_id]
            if len(member_indices) < min_points:
                continue
            instances.append(Instance(next_id, int(semantic_class), member_indices,
                                      cloud.select(np.isin(np.arange(len(cloud.points)), member_indices))))
            next_id += 1
    return instances
