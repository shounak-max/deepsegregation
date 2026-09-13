"""Segmentation and geometry evaluation metrics."""

from __future__ import annotations

import numpy as np


def segmentation_metrics(targets: np.ndarray, predictions: np.ndarray, num_classes: int = 3):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)
    np.add.at(confusion, (targets[valid], predictions[valid]), 1)
    tp = np.diag(confusion).astype(float)
    recall = np.divide(tp, confusion.sum(axis=1), out=np.zeros(num_classes), where=confusion.sum(axis=1) > 0)
    iou = np.divide(tp, confusion.sum(axis=1) + confusion.sum(axis=0) - tp,
                    out=np.zeros(num_classes), where=(confusion.sum(axis=1) + confusion.sum(axis=0) - tp) > 0)
    f1 = np.divide(2*tp, confusion.sum(axis=1) + confusion.sum(axis=0),
                   out=np.zeros(num_classes), where=(confusion.sum(axis=1) + confusion.sum(axis=0)) > 0)
    return {"accuracy": float(tp.sum() / max(1, valid.sum())), "mIoU": float(iou.mean()),
            "per_class_recall": recall.tolist(), "per_class_iou": iou.tolist(),
            "macro_f1": float(f1.mean()), "confusion_matrix": confusion.tolist()}


def relative_error(estimated: float, nominal: float) -> float:
    if nominal <= 0:
        raise ValueError("nominal must be positive")
    return abs(estimated - nominal) / nominal


def geometry_metrics(estimated: np.ndarray, nominal: np.ndarray):
    estimated, nominal = np.asarray(estimated, float), np.asarray(nominal, float)
    if estimated.shape != nominal.shape:
        raise ValueError("estimated and nominal arrays must match")
    errors = np.abs(estimated - nominal) / np.maximum(np.abs(nominal), 1e-12)
    return {"MRE": float(errors.mean()), "max_relative_error": float(errors.max()),
            "per_item_relative_error": errors.tolist()}
