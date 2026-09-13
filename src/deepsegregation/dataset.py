"""PSNet-compatible area discovery and label remapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from .pointcloud import PointCloud, read_ply


@dataclass(frozen=True)
class AreaSample:
    area: str
    path: Path
    cloud: PointCloud


def iter_area_samples(root: str | Path) -> Iterator[AreaSample]:
    root = Path(root)
    areas = sorted(p for p in root.glob("Area_*") if p.is_dir())
    if not areas:
        raise FileNotFoundError(f"no Area_N directories found under {root}")
    for area in areas:
        files = sorted(area.glob("*.ply"))
        for path in files:
            yield AreaSample(area.name, path, read_ply(path))


def remap_labels(labels: np.ndarray, mapping: Mapping[int, int], unknown: int = 0) -> np.ndarray:
    values = np.asarray(labels)
    output = np.full(values.shape, unknown, dtype=np.int64)
    for source, target in mapping.items():
        output[values == source] = target
    return output


def to_psnet_arrays(sample: AreaSample, label_mapping: Mapping[int, int] | None = None):
    if sample.cloud.labels is None:
        raise ValueError(f"{sample.path} has no labels")
    labels = sample.cloud.labels if label_mapping is None else remap_labels(sample.cloud.labels, label_mapping)
    features = sample.cloud.points.astype(np.float32)
    if sample.cloud.normals is not None:
        features = np.column_stack((features, sample.cloud.normals.astype(np.float32)))
    return features, labels.astype(np.int64)
