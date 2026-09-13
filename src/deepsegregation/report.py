"""Compliance calculations and JSON-serializable result records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass(frozen=True)
class ComplianceResult:
    pipe_id: str
    geometry: str
    estimated_radius: float
    nominal_radius: Optional[float]
    absolute_error: Optional[float]
    relative_error: Optional[float]
    tolerance: float
    passed: Optional[bool]
    bend_radius: Optional[float] = None
    bend_angle: Optional[float] = None


def compliance(pipe_id: str, geometry: str, estimated_radius: float,
               nominal_radius: Optional[float], tolerance: float = .05,
               bend_radius: Optional[float] = None,
               bend_angle: Optional[float] = None) -> ComplianceResult:
    if estimated_radius <= 0 or (nominal_radius is not None and nominal_radius <= 0):
        raise ValueError("radii must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if nominal_radius is None:
        return ComplianceResult(pipe_id, geometry, estimated_radius, None, None, None,
                                tolerance, None, bend_radius, bend_angle)
    absolute = abs(estimated_radius - nominal_radius)
    relative = absolute / nominal_radius
    return ComplianceResult(pipe_id, geometry, estimated_radius, nominal_radius,
                            absolute, relative, tolerance, relative <= tolerance,
                            bend_radius, bend_angle)


def write_report(path: str | Path, results: list[ComplianceResult], metadata=None) -> None:
    payload = {"metadata": metadata or {}, "pipes": [asdict(result) for result in results]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
