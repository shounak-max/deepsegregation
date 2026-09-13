"""Torch-lazy inference bridge for the three-class ResPointNet++ head."""

from __future__ import annotations

import numpy as np


def predict_labels(model, features: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Run any compatible torch model and return one semantic label per point.

    The adapter accepts models returning either logits directly or a mapping
    containing ``logits``/``pred``. It intentionally does not prescribe the
    upstream checkpoint construction, which varies across ResPointNet++ forks.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("model inference requires PyTorch") from exc
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("features must have shape (N, F)")
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(array).to(device)
        output = model(tensor)
        if isinstance(output, dict):
            output = output.get("logits", output.get("pred"))
        if output is None:
            raise ValueError("model output has no logits/pred field")
        if output.ndim != 2 or output.shape[0] != len(array):
            raise ValueError("model output must have shape (N, C)")
        return output.argmax(dim=1).cpu().numpy().astype(np.int64)
