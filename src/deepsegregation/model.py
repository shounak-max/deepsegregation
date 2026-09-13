"""Small reference segmentation model for smoke training.

This is a project-owned baseline, not a replacement for the upstream
ResPointNet++ backbone. It makes the data/loss/checkpoint path executable on
the available K80 host while the pinned upstream stack is being resolved.
"""

from __future__ import annotations


def build_point_mlp(input_features: int = 3, num_classes: int = 3):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PointMLP requires PyTorch") from exc
    if input_features < 3 or num_classes < 2:
        raise ValueError("invalid model dimensions")

    class PointMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_features, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(), nn.Dropout(p=.1),
                nn.Linear(128, num_classes),
            )

        def forward(self, features):
            shape = features.shape
            if features.ndim == 3:
                features = features.reshape(-1, shape[-1])
            if features.ndim != 2 or features.shape[-1] != input_features:
                raise ValueError("features must have shape (N,F) or (B,N,F)")
            logits = self.network(features)
            return logits.reshape(*shape[:-1], num_classes)

    return PointMLP()
