"""Hand-computed fixture and corruption checks for the independent verifier."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.verify_evaluation import verify_report


LEVELS = ("action", "short_term", "long_term")


def fixture():
    predicted = [[0, 0, 0], [1, 1, 1], [1, 0, 1]]
    truth = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    names = {level: [f"{level}-0", f"{level}-1"] for level in LEVELS}
    rows = []
    for index, (p, y) in enumerate(zip(predicted, truth)):
        rows.append({
            "video_id": f"{index:03d}", "instance_id": "person", "labels": y,
            "predictions": dict(zip(LEVELS, p)),
            "probabilities": {level: [0.9, 0.1] if value == 0 else [0.1, 0.9]
                              for level, value in zip(LEVELS, p)},
            "prediction_names": {level: names[level][value] for level, value in zip(LEVELS, p)},
        })
    metrics = {"num_samples": 3, "mean_mAP": 35 / 72, "mean_accuracy": 1 / 9,
               "mean_macro_f1": 1 / 12, "triplet_accuracy": 0.0,
               "consistency": 2 / 3, "S_cons": 2 / 3}
    for level in LEVELS:
        short = level == "short_term"
        metrics.update({
            f"{level}_mAP": 11 / 24 if short else 0.5,
            f"{level}_accuracy": 1 / 3 if short else 0.0,
            f"{level}_macro_f1": 0.25 if short else 0.0,
            f"{level}_class_AP": [7 / 12, 1 / 3] if short else [2 / 3, 1 / 3],
            f"{level}_class_F1": [0.5, 0.0] if short else [0.0, 0.0],
            f"{level}_class_support": [2, 1],
            f"{level}_classes_with_positives": 2,
            f"{level}_num_classes": 2,
        })
    return {"num_instances": 3, "label_names": names, "metrics": metrics}, rows


class IndependentVerificationTests(unittest.TestCase):
    def run_fixture(self, document, rows, compatibility=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metrics.json").write_text(json.dumps(document))
            (path / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (path / "compatibility.json").write_text(json.dumps(
                compatibility if compatibility is not None else {"valid_triplets": [[0, 0, 0], [1, 1, 1]]}))
            return verify_report(path / "metrics.json", path / "predictions.jsonl",
                                 path / "compatibility.json", sklearn_mode="off")

    def test_hand_computed_metrics_and_wrong_but_valid_predictions(self):
        document, rows = fixture()
        result = self.run_fixture(document, rows)
        self.assertTrue(result["verified"])
        self.assertEqual(result["valid_triplets"], 2)
        self.assertEqual(result["wrong_but_valid_triplets"], 2)
        self.assertEqual(result["invalid_triplets"], 1)
        self.assertEqual(result["correct_triplets"], 0)
        self.assertEqual(result["compatibility_valid_triplets"], 2)

    def test_modified_metric_is_detected(self):
        document, rows = fixture()
        document["metrics"]["short_term_class_AP"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "short_term_class_AP"):
            self.run_fixture(document, rows)

    def test_constrained_exported_prediction_is_detected(self):
        document, rows = fixture()
        rows[2]["predictions"]["action"] = 0
        with self.assertRaisesRegex(ValueError, "not the raw argmax"):
            self.run_fixture(document, rows)

    def test_duplicate_instance_is_detected(self):
        document, rows = fixture()
        rows[2]["video_id"] = rows[0]["video_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_fixture(document, rows)

    def test_dense_compatibility_definition(self):
        document, rows = fixture()
        dense = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]]
        self.assertTrue(self.run_fixture(document, rows, {"compatibility": dense})["verified"])


if __name__ == "__main__":
    unittest.main()
