#!/usr/bin/env python3
"""Independently verify exported HINT-Net evaluation from raw probabilities.

Requires only NumPy; it deliberately imports no hintnet code. AP is recomputed
by sweeping distinct score thresholds, F1 from an explicit confusion matrix,
and paper S_cons from the raw, independent argmax triplets. An available sklearn
installation adds a second numerical comparison without being required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np


LEVELS = ("action", "short_term", "long_term")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compatibility(path, sizes):
    document = _read_json(path)
    _require(isinstance(document, dict), "Compatibility must be a JSON object")
    representations = [key for key in ("valid_triplets", "compatibility") if key in document]
    _require(len(representations) == 1, "Compatibility needs exactly one of valid_triplets or compatibility")
    if "valid_triplets" in document:
        triples = document["valid_triplets"]
        _require(isinstance(triples, list), "valid_triplets must be a list")
        matrix = np.zeros(sizes, dtype=bool)
        for index, triple in enumerate(triples):
            _require(isinstance(triple, list) and len(triple) == 3
                     and all(type(value) is int for value in triple),
                     f"Compatibility triplet {index} must contain three integer IDs")
            _require(all(0 <= value < size for value, size in zip(triple, sizes)),
                     f"Compatibility triplet {index} is outside class ranges {sizes}")
            matrix[tuple(triple)] = True
        return matrix
    matrix = np.asarray(document["compatibility"])
    _require(matrix.shape == tuple(sizes), f"Compatibility shape must be {sizes}, got {matrix.shape}")
    _require(matrix.dtype.kind in "biuf" and bool(np.all((matrix == 0) | (matrix == 1))),
             "Compatibility entries must be booleans or exactly 0/1")
    return matrix.astype(bool)


def _threshold_ap(truth, scores):
    """Independent threshold-sweep AP; equal scores enter the set together."""
    positives = int(sum(bool(value) for value in truth))
    if not positives:
        return 0.0
    previous_recall, ap = 0.0, 0.0
    for threshold in sorted(set(float(value) for value in scores), reverse=True):
        selected = scores >= threshold
        true_positive = int(np.count_nonzero(truth & selected))
        recall = true_positive / positives
        precision = true_positive / int(np.count_nonzero(selected))
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def _assert_matches(actual, expected, name, atol):
    _require(not isinstance(actual, bool), f"{name}: boolean is not a numeric metric")
    try:
        first, second = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: metric must be numeric") from exc
    _require(first.shape == second.shape, f"{name}: shape {first.shape} differs from recomputed {second.shape}")
    _require(bool(np.isfinite(first).all()), f"{name}: metric contains non-finite values")
    _require(bool(np.allclose(first, second, rtol=0, atol=atol)),
             f"{name}: reported {actual!r} differs from independently recomputed {expected!r}")


def verify_report(metrics_path, predictions_path, compatibility_path, labels_path=None,
                  *, atol=1e-10, sklearn_mode="auto"):
    """Return a JSON-serializable verification result, raising on any mismatch."""
    _require(np.isfinite(atol) and atol >= 0, "atol must be finite and nonnegative")
    _require(sklearn_mode in ("auto", "required", "off"), "Unknown sklearn mode")
    document = _read_json(metrics_path)
    _require(isinstance(document, dict) and isinstance(document.get("metrics"), dict),
             "Evaluation JSON must contain a metrics object")
    reported = document["metrics"]
    rows = []
    for line_number, line in enumerate(Path(predictions_path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Prediction line {line_number}: invalid JSON") from exc
            _require(isinstance(row, dict), f"Prediction line {line_number} must be an object")
            rows.append(row)
    _require(bool(rows), "Predictions are empty")
    sizes = []
    for level in LEVELS:
        values = rows[0].get("probabilities", {}).get(level)
        _require(isinstance(values, list) and len(values) > 0,
                 f"First prediction needs a nonempty probability list for {level}")
        sizes.append(len(values))
    names = _read_json(labels_path) if labels_path else document.get("label_names")
    if names is not None:
        _require(isinstance(names, dict), "Label names must be an object keyed by semantic level")
        for level, size in zip(LEVELS, sizes):
            values = names.get(level)
            _require(isinstance(values, list) and len(values) == size
                     and all(isinstance(value, str) for value in values),
                     f"Label names for {level} must contain exactly {size} strings")
            if labels_path and "label_names" in document:
                _require(values == document["label_names"].get(level),
                         f"Supplied {level} label names differ from evaluation JSON")
    truth, probabilities = [], {level: [] for level in LEVELS}
    identifiers = set()
    for index, row in enumerate(rows):
        prefix = f"Prediction row {index}"
        labels = row.get("labels")
        _require(isinstance(labels, list) and len(labels) == 3
                 and all(type(value) is int for value in labels),
                 f"{prefix}: three integer ground-truth labels are required")
        _require(all(0 <= value < size for value, size in zip(labels, sizes)),
                 f"{prefix}: labels are outside output class ranges")
        identifier = (row.get("video_id"), row.get("instance_id"))
        _require(all(isinstance(value, str) and value for value in identifier),
                 f"{prefix}: nonempty video_id and instance_id strings are required")
        _require(identifier not in identifiers, f"{prefix}: duplicate video/instance {identifier}")
        identifiers.add(identifier)
        truth.append(labels)
        for level, size in zip(LEVELS, sizes):
            values = np.asarray(row.get("probabilities", {}).get(level), dtype=np.float64)
            _require(values.shape == (size,) and bool(np.isfinite(values).all())
                     and bool(np.all((values >= 0) & (values <= 1)))
                     and bool(np.isclose(values.sum(), 1, atol=1e-4, rtol=1e-4)),
                     f"{prefix}/{level}: malformed probability distribution")
            predicted = int(values.argmax())
            exported = row.get("predictions", {}).get(level)
            _require(type(exported) is int and exported == predicted,
                     f"{prefix}/{level}: exported prediction is not the raw argmax")
            if names is not None and "prediction_names" in row:
                _require(row["prediction_names"].get(level) == names[level][predicted],
                         f"{prefix}/{level}: prediction name differs from the class-ID vocabulary")
            probabilities[level].append(values)
    y = np.asarray(truth, dtype=np.int64)
    recomputed = {"num_samples": len(rows)}
    predictions = []
    for column, (level, size) in enumerate(zip(LEVELS, sizes)):
        scores = np.asarray(probabilities[level])
        predicted = scores.argmax(axis=1)
        predictions.append(predicted)
        confusion = np.zeros((size, size), dtype=np.int64)
        for actual, inferred in zip(y[:, column], predicted):
            confusion[actual, inferred] += 1
        support = confusion.sum(axis=1).tolist()
        class_f1 = []
        for class_id in range(size):
            tp = int(confusion[class_id, class_id])
            fp = int(confusion[:, class_id].sum()) - tp
            fn = int(confusion[class_id].sum()) - tp
            class_f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        class_ap = [_threshold_ap(y[:, column] == class_id, scores[:, class_id])
                    for class_id in range(size)]
        recomputed.update({
            f"{level}_mAP": sum(class_ap) / size,
            f"{level}_accuracy": int(np.trace(confusion)) / len(rows),
            f"{level}_macro_f1": sum(class_f1) / size,
            f"{level}_class_AP": class_ap,
            f"{level}_class_F1": class_f1,
            f"{level}_class_support": support,
            f"{level}_classes_with_positives": sum(value > 0 for value in support),
            f"{level}_num_classes": size,
        })
    for name in ("mAP", "accuracy", "macro_f1"):
        recomputed[f"mean_{name}"] = sum(recomputed[f"{level}_{name}"] for level in LEVELS) / 3
    predicted_triplets = np.stack(predictions, axis=1)
    correct = np.all(predicted_triplets == y, axis=1)
    matrix = _compatibility(compatibility_path, sizes)
    valid = matrix[tuple(predictions)]
    recomputed["triplet_accuracy"] = int(correct.sum()) / len(rows)
    recomputed["S_cons"] = int(valid.sum()) / len(rows)
    recomputed["consistency"] = recomputed["S_cons"]
    for name, value in recomputed.items():
        _require(name in reported, f"Evaluation metric is missing: {name}")
        _assert_matches(reported[name], value, name, atol)
    if "num_instances" in document:
        _require(document["num_instances"] == len(rows), "num_instances differs from exported prediction count")
    sklearn_report = {"status": "disabled" if sklearn_mode == "off" else "not_installed"}
    if sklearn_mode != "off":
        available = importlib.util.find_spec("sklearn") is not None
        _require(available or sklearn_mode != "required", "sklearn crosscheck requested but sklearn is not installed")
        if available:
            import sklearn
            from sklearn.metrics import accuracy_score, average_precision_score, f1_score
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for column, (level, size) in enumerate(zip(LEVELS, sizes)):
                    scores = np.asarray(probabilities[level])
                    ap = [float(average_precision_score(y[:, column] == class_id, scores[:, class_id]))
                          for class_id in range(size)]
                    f1 = f1_score(y[:, column], predictions[column], labels=np.arange(size),
                                  average=None, zero_division=0).tolist()
                    _assert_matches(recomputed[f"{level}_class_AP"], ap, f"sklearn/{level}/AP", atol)
                    _assert_matches(recomputed[f"{level}_class_F1"], f1, f"sklearn/{level}/F1", atol)
                    _assert_matches(recomputed[f"{level}_accuracy"], float(accuracy_score(y[:, column], predictions[column])),
                                    f"sklearn/{level}/accuracy", atol)
            sklearn_report = {"status": "verified", "version": sklearn.__version__}
    return {
        "verified": True,
        "method": "Independent NumPy threshold AP, explicit confusion F1, and raw-argmax paper S_cons; no hintnet imports",
        "metrics_file": str(Path(metrics_path).resolve()),
        "predictions_file": str(Path(predictions_path).resolve()),
        "compatibility_file": str(Path(compatibility_path).resolve()),
        "num_samples": len(rows), "class_counts": dict(zip(LEVELS, sizes)),
        "metric_keys_checked": len(recomputed), "absolute_tolerance": atol,
        "compatibility_valid_triplets": int(matrix.sum()),
        "valid_triplets": int(valid.sum()), "invalid_triplets": int((~valid).sum()),
        "correct_triplets": int(correct.sum()),
        "wrong_but_valid_triplets": int((valid & ~correct).sum()),
        "S_cons": recomputed["S_cons"], "triplet_accuracy": recomputed["triplet_accuracy"],
        "mean_mAP": recomputed["mean_mAP"], "mean_accuracy": recomputed["mean_accuracy"],
        "mean_macro_f1": recomputed["mean_macro_f1"],
        "sklearn_crosscheck": sklearn_report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--compatibility", required=True, type=Path)
    parser.add_argument("--labels", type=Path, help="optional class-ID vocabulary; otherwise use evaluation label_names")
    parser.add_argument("--output", type=Path, help="optional verification JSON")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--sklearn", choices=("auto", "required", "off"), default="auto")
    args = parser.parse_args()
    try:
        result = verify_report(args.metrics, args.predictions, args.compatibility, args.labels,
                               atol=args.atol, sklearn_mode=args.sklearn)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"verified": False, "error": str(exc)}
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
