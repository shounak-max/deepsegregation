"""Small, dependency-light point-cloud container and ASCII PLY I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if not np.isfinite(self.points).all():
            raise ValueError("points must contain finite values")
        for name in ("colors", "normals", "labels"):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value)
            if value.shape[0] != len(self.points):
                raise ValueError(f"{name} must have one row per point")
            if name != "labels" and (value.ndim != 2 or value.shape[1] != 3):
                raise ValueError(f"{name} must have shape (N, 3)")
            setattr(self, name, value)

    def select(self, mask: np.ndarray) -> "PointCloud":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self.points),):
            raise ValueError("mask must have one entry per point")
        return PointCloud(
            self.points[mask],
            None if self.colors is None else self.colors[mask],
            None if self.normals is None else self.normals[mask],
            None if self.labels is None else self.labels[mask],
            dict(self.metadata),
        )


def _property_names(header):
    names = []
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
            names.append(parts[2])
    return names


def read_ply(path: str | Path) -> PointCloud:
    """Read an ASCII PLY with x/y/z and optional color, normal, label fields."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        header = []
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            line = line.strip()
            if line == "end_header":
                break
            header.append(line)
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
        if vertex_count is None:
            raise ValueError("PLY has no vertex element")
        names = _property_names(header)
        data = np.loadtxt(handle, dtype=float, max_rows=vertex_count)
    data = np.atleast_2d(data)
    if data.shape[1] != len(names):
        raise ValueError("PLY vertex data does not match its header")
    columns = {name: data[:, i] for i, name in enumerate(names)}
    required = {"x", "y", "z"}
    if not required.issubset(columns):
        raise ValueError("PLY must contain x, y, and z properties")
    points = np.column_stack([columns[name] for name in ("x", "y", "z")])
    colors = np.column_stack([columns[name] for name in ("red", "green", "blue")]) if {"red", "green", "blue"}.issubset(columns) else None
    normals = np.column_stack([columns[name] for name in ("nx", "ny", "nz")]) if {"nx", "ny", "nz"}.issubset(columns) else None
    label_name = next((name for name in ("label", "class", "semantic") if name in columns), None)
    labels = columns[label_name].astype(np.int64) if label_name else None
    return PointCloud(points, colors, normals, labels)


def write_ply(path: str | Path, cloud: PointCloud) -> None:
    """Write a portable ASCII PLY, retaining fields present on ``cloud``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [("x", "double"), ("y", "double"), ("z", "double")]
    arrays = [cloud.points]
    if cloud.normals is not None:
        fields.extend((name, "double") for name in ("nx", "ny", "nz"))
        arrays.append(cloud.normals)
    if cloud.colors is not None:
        fields.extend((name, "uchar") for name in ("red", "green", "blue"))
        arrays.append(np.clip(cloud.colors, 0, 255).astype(np.uint8))
    if cloud.labels is not None:
        fields.append(("label", "int"))
        arrays.append(np.asarray(cloud.labels, dtype=np.int64).reshape(-1, 1))
    matrix = np.column_stack(arrays)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(cloud.points)}\n")
        for name, kind in fields:
            handle.write(f"property {kind} {name}\n")
        handle.write("end_header\n")
        for row in matrix:
            handle.write(" ".join(str(int(v)) if kind in ("uchar", "int") else f"{v:.9g}" for v, (_, kind) in zip(row, fields)) + "\n")


def write_json(path: str | Path, payload: Dict[str, object]) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
