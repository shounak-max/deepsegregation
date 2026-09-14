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
    target_support = confusion.sum(axis=1)
    prediction_support = confusion.sum(axis=0)
    union = target_support + prediction_support - tp
    recall = np.divide(tp, target_support, out=np.zeros(num_classes), where=target_support > 0)
    precision = np.divide(tp, prediction_support, out=np.zeros(num_classes), where=prediction_support > 0)
    iou = np.divide(tp, union, out=np.zeros(num_classes), where=union > 0)
    f1 = np.divide(2 * tp, target_support + prediction_support,
                   out=np.zeros(num_classes), where=(target_support + prediction_support) > 0)

    # A class absent from the labels has no measurable recall or IoU.  Averaging
    # it in as a hard zero makes a real two-class evaluation look worse merely
    # because the model's output head also supports a third class.  Keep the
    # all-configured-class values below for comparison, but make the headline
    # metric explicit about the classes actually evaluated by ground truth.
    evaluated = target_support > 0
    supported_miou = float(iou[evaluated].mean()) if np.any(evaluated) else 0.0
    supported_macro_f1 = float(f1[evaluated].mean()) if np.any(evaluated) else 0.0

    return {
        "accuracy": float(tp.sum() / max(1, valid.sum())),
        "mIoU": supported_miou,
        "mIoU_all_classes": float(iou.mean()),
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_iou": iou.tolist(),
        "per_class_target_support": target_support.tolist(),
        "per_class_prediction_support": prediction_support.tolist(),
        "evaluated_class_indices": np.flatnonzero(evaluated).tolist(),
        "per_class_status": ["evaluated" if present else "undefined_no_ground_truth"
                             for present in evaluated.tolist()],
        "macro_f1": supported_macro_f1,
        "macro_f1_all_classes": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }


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
