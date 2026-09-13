"""End-to-end orchestration boundary for segmentation-to-report inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from .clustering import Instance, separate_instances
from .cylinder import CylinderFit, fit_cylinder_ransac
from .pointcloud import PointCloud
from .preprocessing import preprocess
from .report import ComplianceResult, compliance
from .torus import TorusFit, fit_torus_ransac


@dataclass
class PipelineResult:
    processed_cloud: PointCloud
    instances: list[Instance]
    fits: dict[int, CylinderFit | TorusFit] = field(default_factory=dict)
    compliance: list[ComplianceResult] = field(default_factory=list)


def run_pipeline(cloud: PointCloud, semantic_labels: Optional[np.ndarray] = None,
                 *, voxel_size: Optional[float] = None, dbscan_eps: float = .05,
                 dbscan_min_samples: int = 12, fit_threshold: float = .01,
                 nominal_radii: Optional[Mapping[int, float]] = None,
                 tolerance: float = .05) -> PipelineResult:
    working = cloud if semantic_labels is None else PointCloud(
        cloud.points, cloud.colors, cloud.normals, np.asarray(semantic_labels), dict(cloud.metadata)
    )
    processed = preprocess(working, voxel_size=voxel_size, denoise=False, compute_normals=False)
    labels = processed.labels
    if labels is None:
        raise ValueError("semantic_labels or cloud.labels is required")
    if len(labels) != len(processed.points):
        raise ValueError("semantic labels must match the processed point count")
    instances = separate_instances(processed, labels, eps=dbscan_eps,
                                   min_samples=dbscan_min_samples)
    fits = {}
    reports = []
    for instance in instances:
        try:
            if instance.semantic_class == 2:
                fit = fit_torus_ransac(instance.cloud.points, distance_threshold=fit_threshold,
                                       min_inliers=max(6, int(.4 * len(instance.cloud.points))))
                estimate = fit.minor_radius
                reports.append(compliance(str(instance.instance_id), "elbow", estimate,
                                          None if nominal_radii is None else nominal_radii.get(instance.instance_id),
                                          tolerance, bend_radius=fit.major_radius))
            else:
                fit = fit_cylinder_ransac(instance.cloud.points, distance_threshold=fit_threshold,
                                          min_inliers=max(6, int(.4 * len(instance.cloud.points))))
                estimate = fit.radius
                reports.append(compliance(str(instance.instance_id), "straight", estimate,
                                          None if nominal_radii is None else nominal_radii.get(instance.instance_id),
                                          tolerance))
            fits[instance.instance_id] = fit
        except (RuntimeError, ValueError):
            # Preserve the separated instance for diagnostics even when its
            # geometry is too incomplete for a reliable fit.
            continue
    return PipelineResult(processed, instances, fits, reports)
