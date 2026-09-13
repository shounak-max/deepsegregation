"""Three-class segmentation contracts and Boundary-CB loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SegmentationConfig:
    num_classes: int = 3
    class_names: tuple = ("background", "pipe", "elbow")
    boundary_weight: float = 2.0
    class_balance_beta: float = 0.999

    def __post_init__(self):
        if self.num_classes != len(self.class_names):
            raise ValueError("num_classes must match class_names")
        if self.boundary_weight < 1 or not 0 <= self.class_balance_beta < 1:
            raise ValueError("invalid Boundary-CB parameters")


def boundary_mask_from_labels(points: np.ndarray, labels: np.ndarray, neighbors: int = 8) -> np.ndarray:
    """Mark points whose neighborhood contains another semantic class."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    if len(points) != len(labels):
        raise ValueError("points and labels must have equal length")
    from scipy.spatial import cKDTree
    _, indices = cKDTree(points).query(points, k=min(neighbors + 1, len(points)))
    return np.any(labels[indices[:, 1:]] != labels[:, None], axis=1)


def boundary_class_balanced_loss(logits, targets, boundary_mask=None,
                                 beta: float = 0.999, boundary_weight: float = 2.0):
    """Compute class-balanced cross entropy with extra boundary emphasis.

    This function is kept torch-lazy so dataset and geometry tools work on
    machines without PyTorch. It accepts logits shaped ``(N, C)`` and integer
    targets shaped ``(N,)``.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Boundary-CB loss requires PyTorch") from exc
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must be (N,C) and targets must be (N,)")
    if not 0 <= beta < 1 or boundary_weight < 1:
        raise ValueError("invalid loss parameters")
    classes = logits.shape[1]
    counts = torch.bincount(targets, minlength=classes).float()
    effective = 1.0 - torch.pow(torch.tensor(beta, device=logits.device), counts)
    weights = torch.where(counts > 0, (1.0 - beta) / effective, torch.zeros_like(counts))
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps) * classes
    loss = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    if boundary_mask is not None:
        boundary_mask = boundary_mask.to(dtype=loss.dtype)
        loss = loss * (1.0 + (boundary_weight - 1.0) * boundary_mask)
    return loss.mean()
