"""Evaluation and ablation helpers for the roadmap's promised result tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import time

import numpy as np

from .metrics import geometry_metrics, segmentation_metrics


@dataclass(frozen=True)
class AblationRow:
    name: str
    segmentation: dict
    geometry: dict
    processing_seconds: float


def evaluate_ablation(cases: Iterable[tuple[str, Callable]], targets: np.ndarray,
                      nominal_radii: np.ndarray | None = None) -> list[AblationRow]:
    """Run named RANSAC-only/DL-only/hybrid callables on identical inputs."""
    rows = []
    for name, operation in cases:
        start = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - start
        if isinstance(result, tuple):
            predictions, estimates = result
        else:
            predictions, estimates = result, None
        segmentation = segmentation_metrics(targets, predictions, int(max(targets.max(), predictions.max()) + 1))
        geometry = {} if estimates is None or nominal_radii is None else geometry_metrics(estimates, nominal_radii)
        rows.append(AblationRow(name, segmentation, geometry, elapsed))
    return rows
