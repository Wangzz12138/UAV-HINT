"""Probability-based macro AP and raw-argmax hierarchical consistency."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import Tensor

from .objectives import TASKS, _compatibility_tensor


def _numpy(value) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        # NumPy cannot represent bfloat16. Preserve float64 scores and their ties.
        return (value.float() if value.dtype == torch.bfloat16 else value).numpy()
    return np.asarray(value)


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """Non-interpolated AP, grouping equal scores at a common threshold.

    This agrees with the binary average_precision_score convention. A class with
    no positives contributes zero, so macro AP always averages all declared classes.
    """
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    ordered_score = score[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(ordered_score)), len(order) - 1]
    true_positives = np.cumsum(target[order], dtype=np.float64)[threshold_ends]
    precision = true_positives / (threshold_ends + 1)
    recall_increments = np.diff(np.r_[0.0, true_positives]) / positives
    return float(np.dot(recall_increments, precision))


def evaluate_predictions(
    probabilities: Mapping[str, Tensor | np.ndarray],
    labels: Tensor | np.ndarray,
    compatibility: Optional[Tensor | np.ndarray] = None,
) -> dict[str, float | int | list[float] | list[int]]:
    """Evaluate three softmax probability matrices and integer labels ``[N,3]``.

    AP is one-vs-rest per class and macro-averaged over every output class, with
    absent-positive classes assigned AP=0. Class coverage is returned alongside
    each mAP to expose incomplete evaluation subsets. Fractions use [0,1], not %.
    Per-class AP, F1 and support lists follow the output class-ID order. Macro-F1
    averages every declared class, with undefined F1 assigned zero. The mean
    accuracy and mean macro-F1 average the three heads equally; triplet accuracy
    instead requires all three labels to be correct for the same sample.
    ``S_cons`` and the backwards-compatible ``consistency`` key are identical:
    mean compatibility of independent argmax triplets (paper equations 1–2).
    Both are omitted if no explicit compatibility definition is provided.
    This function never applies constrained decoding to the independent argmaxes.
    """
    y = _numpy(labels)
    if y.ndim != 2 or y.shape[1] != 3 or y.shape[0] == 0:
        raise ValueError("labels must have shape [nonempty samples, 3]")
    if not np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
        raise ValueError("labels must contain integer class IDs")
    result: dict[str, float | int | list[float] | list[int]] = {"num_samples": int(y.shape[0])}
    predicted = []
    class_counts = []
    for index, task in enumerate(TASKS):
        if task not in probabilities:
            raise ValueError(f"missing probabilities for {task}")
        values = _numpy(probabilities[task])
        if values.ndim != 2 or values.shape[0] != y.shape[0] or values.shape[1] == 0:
            raise ValueError(f"{task} probabilities must have shape [{y.shape[0]}, nonempty classes]")
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError(f"{task} probabilities must contain finite floating-point values")
        if np.any((values < 0) | (values > 1)) or not np.allclose(values.sum(axis=1), 1, atol=1e-4, rtol=1e-4):
            raise ValueError(f"{task} values must be probabilities in [0,1] with each row summing to 1")
        classes = values.shape[1]
        if np.any((y[:, index] < 0) | (y[:, index] >= classes)):
            raise ValueError(f"labels for {task} are outside [0, {classes})")
        aps = [_average_precision(y[:, index] == class_id, values[:, class_id]) for class_id in range(classes)]
        prediction = values.argmax(axis=1)
        support = np.bincount(y[:, index], minlength=classes)
        predicted_support = np.bincount(prediction, minlength=classes)
        true_positives = np.bincount(y[:, index][prediction == y[:, index]], minlength=classes)
        denominator = support + predicted_support
        class_f1 = np.divide(2.0 * true_positives, denominator,
                             out=np.zeros(classes, dtype=np.float64), where=denominator != 0)
        predicted.append(prediction)
        class_counts.append(classes)
        result[f"{task}_mAP"] = float(np.mean(aps))
        result[f"{task}_accuracy"] = float(np.mean(prediction == y[:, index]))
        result[f"{task}_macro_f1"] = float(np.mean(class_f1))
        result[f"{task}_class_AP"] = aps
        result[f"{task}_class_F1"] = class_f1.tolist()
        result[f"{task}_class_support"] = support.tolist()
        result[f"{task}_classes_with_positives"] = int(np.unique(y[:, index]).size)
        result[f"{task}_num_classes"] = int(classes)
    result["mean_mAP"] = float(np.mean([result[f"{task}_mAP"] for task in TASKS]))
    result["mean_accuracy"] = float(np.mean([result[f"{task}_accuracy"] for task in TASKS]))
    result["mean_macro_f1"] = float(np.mean([result[f"{task}_macro_f1"] for task in TASKS]))
    result["triplet_accuracy"] = float(np.all(np.stack(predicted, axis=1) == y, axis=1).mean())
    if compatibility is not None:
        matrix = _compatibility_tensor(compatibility, class_counts).cpu().numpy()
        result["consistency"] = float(np.mean(matrix[tuple(predicted)]))
        result["S_cons"] = result["consistency"]
    return result
