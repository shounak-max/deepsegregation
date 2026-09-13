"""Synthetic pipe scans with exact geometry ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .pointcloud import PointCloud
from .pointcloud import write_json, write_ply


BACKGROUND, STRAIGHT_PIPE, ELBOW = 0, 1, 2


@dataclass(frozen=True)
class SyntheticScan:
    cloud: PointCloud
    ground_truth: Dict[str, float]


def generate_elbow(major_radius: float = 0.75, minor_radius: float = 0.08,
                   bend_angle: float = np.pi / 2, samples: int = 2400,
                   gaussian_noise: float = 0.0, dropout: float = 0.0,
                   outliers: int = 0, seed: int = 0) -> SyntheticScan:
    if major_radius <= minor_radius or minor_radius <= 0:
        raise ValueError("major_radius must exceed positive minor_radius")
    if not 0 < bend_angle <= np.pi:
        raise ValueError("bend_angle must be in (0, pi]")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-bend_angle / 2, bend_angle / 2, samples)
    phi = rng.uniform(0, 2 * np.pi, samples)
    points = np.column_stack(((major_radius + minor_radius * np.cos(phi)) * np.cos(theta),
                              (major_radius + minor_radius * np.cos(phi)) * np.sin(theta),
                              minor_radius * np.sin(phi)))
    labels = np.full(samples, ELBOW, dtype=np.int64)
    if gaussian_noise:
        points += rng.normal(scale=gaussian_noise, size=points.shape)
    keep = rng.random(samples) >= dropout
    points, labels = points[keep], labels[keep]
    if outliers:
        clutter = rng.uniform(-1.5 * major_radius, 1.5 * major_radius, (outliers, 3))
        points = np.vstack((points, clutter))
        labels = np.concatenate((labels, np.full(outliers, BACKGROUND, dtype=np.int64)))
    return SyntheticScan(PointCloud(points, labels=labels), {
        "bend_radius": float(major_radius), "pipe_radius": float(minor_radius),
        "bend_angle": float(bend_angle),
    })


def generate_scene(seed: int = 0) -> SyntheticScan:
    """Generate a small two-instance scene for clustering and fitting tests."""
    rng = np.random.default_rng(seed)
    left = generate_elbow(0.7, 0.07, np.pi / 2, 1400, gaussian_noise=.001, seed=seed)
    straight_theta = rng.uniform(0, 1, 900)
    phi = rng.uniform(0, 2 * np.pi, 900)
    straight = np.column_stack((straight_theta * 1.2 - .6,
                                .45 + .07 * np.cos(phi), .07 * np.sin(phi)))
    points = np.vstack((left.cloud.points, straight))
    labels = np.concatenate((left.cloud.labels, np.full(len(straight), STRAIGHT_PIPE, dtype=np.int64)))
    return SyntheticScan(PointCloud(points, labels=labels), {"bend_radius": .7, "pipe_radius": .07})


def write_synthetic_dataset(root: str | Path, count: int = 200, seed: int = 0) -> Path:
    """Generate PSNet-shaped ``Area_N`` scans and JSON geometry manifests."""
    if count < 1:
        raise ValueError("count must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        scan = generate_elbow(
            major_radius=float(rng.uniform(.35, 1.5)),
            minor_radius=float(rng.uniform(.025, .15)),
            bend_angle=float(rng.choice([np.pi / 4, np.pi / 2])),
            samples=1800,
            gaussian_noise=float(rng.uniform(0, .004)),
            dropout=float(rng.uniform(0, .2)),
            outliers=int(rng.uniform(0, 80)),
            seed=seed + index,
        )
        area = root / f"Area_{index + 1}"
        write_ply(area / "scan.ply", scan.cloud)
        write_json(area / "scan.json", {"scan": "scan.ply", **scan.ground_truth})
    return root
