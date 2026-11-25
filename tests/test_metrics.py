"""Independent numerical cases for per-label AP and all-class macro-F1."""

import json
import unittest

import numpy as np

from hintnet.metrics import evaluate_predictions
from hintnet.objectives import TASKS


class PerLabelMetricTests(unittest.TestCase):
    def test_nontrivial_confusion_macro_f1_and_class_order(self):
        # True-row/predicted-column confusion is [[2,1,0],[0,1,1],[1,0,0]].
        truth = np.array([0, 0, 0, 1, 1, 2])
        predicted = np.array([0, 0, 1, 1, 2, 0])
        values = np.full((len(truth), 3), 0.1)
        values[np.arange(len(truth)), predicted] = 0.8
        labels = np.repeat(truth[:, None], 3, axis=1)
        result = evaluate_predictions({task: values.copy() for task in TASKS}, labels)
        for task in TASKS:
            self.assertEqual(result[f"{task}_class_support"], [3, 2, 1])
            np.testing.assert_allclose(result[f"{task}_class_F1"], [2 / 3, 1 / 2, 0])
            self.assertAlmostEqual(result[f"{task}_macro_f1"], 7 / 18)
            self.assertEqual(result[f"{task}_accuracy"], 0.5)
            self.assertEqual(len(result[f"{task}_class_AP"]), 3)
            self.assertEqual(result[f"{task}_mAP"], np.mean(result[f"{task}_class_AP"]))
        self.assertAlmostEqual(result["mean_macro_f1"], 7 / 18)
        self.assertEqual(result["mean_accuracy"], 0.5)
        # Every value must support ordinary JSON serialization without NaN.
        json.dumps(result, allow_nan=False)

    def test_absent_and_never_predicted_classes_contribute_zero(self):
        truth = np.array([0, 0, 1, 1])
        # Always predict class zero; class two is absent from truth and predictions.
        values = np.tile([0.8, 0.15, 0.05], (4, 1))
        labels = np.repeat(truth[:, None], 3, axis=1)
        result = evaluate_predictions({task: values.copy() for task in TASKS}, labels)
        self.assertEqual(result["action_class_support"], [2, 2, 0])
        self.assertEqual(result["action_class_AP"], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(result["action_class_F1"], [2 / 3, 0, 0])
        self.assertAlmostEqual(result["action_macro_f1"], 2 / 9)
        self.assertAlmostEqual(result["action_mAP"], 1 / 3)

    def test_per_class_ap_preserves_ranked_probability_semantics(self):
        values = np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
        truth = np.array([0, 1, 0, 1])
        labels = np.repeat(truth[:, None], 3, axis=1)
        result = evaluate_predictions({task: values.copy() for task in TASKS}, labels)
        np.testing.assert_allclose(result["action_class_AP"], [5 / 6, 5 / 6])
        self.assertAlmostEqual(result["action_mAP"], np.mean(result["action_class_AP"]))

    def test_head_mean_and_triplet_accuracy_have_distinct_meanings(self):
        labels = np.zeros((3, 3), dtype=np.int64)
        probabilities = {}
        for index, task in enumerate(TASKS):
            values = np.tile([0.9, 0.1], (3, 1))
            values[index] = [0.1, 0.9]
            probabilities[task] = values
        result = evaluate_predictions(probabilities, labels)
        self.assertAlmostEqual(result["mean_accuracy"], 2 / 3)
        self.assertEqual(result["triplet_accuracy"], 0.0)
        self.assertEqual(result["mean_macro_f1"], np.mean([result[f"{task}_macro_f1"] for task in TASKS]))

    def test_paper_consistency_counts_valid_triplets_independently_of_truth(self):
        predictions = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1]])
        probabilities = {}
        for index, task in enumerate(TASKS):
            values = np.full((3, 2), 0.1)
            values[np.arange(3), predictions[:, index]] = 0.9
            probabilities[task] = values
        compatibility = np.zeros((2, 2, 2), dtype=bool)
        compatibility[0, 0, 0] = True
        compatibility[1, 1, 1] = True
        # Every predicted triplet is wrong, but exactly two are logically valid.
        labels = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        result = evaluate_predictions(probabilities, labels, compatibility)
        self.assertEqual(result["S_cons"], 2 / 3)
        self.assertEqual(result["S_cons"], result["consistency"])
        self.assertEqual(result["triplet_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
