"""Numerical checks of paper LVP, tied AP, explicit compatibility, and gradients."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hintnet.metrics import evaluate_predictions
from hintnet.objectives import TASKS, HierarchicalLoss, load_compatibility


class LogicalLossTests(unittest.TestCase):
    def test_joint_penalty_exact_value_and_gradient(self):
        compatibility = torch.zeros(2, 2, 2, dtype=torch.bool)
        compatibility[0, 0, 0] = True
        logits = {task: torch.zeros(1, 2, requires_grad=True) for task in TASKS}
        labels = torch.tensor([[0, 0, 0]])
        result = HierarchicalLoss(compatibility, 5)(logits, labels)
        self.assertAlmostEqual(result["lvp"].item(), 7 / 8)
        self.assertAlmostEqual(result["classification"].item(), 3 * np.log(2), places=6)
        self.assertAlmostEqual(result["total"].item(), 3 * np.log(2) + 5 * 7 / 8, places=6)
        result["lvp"].backward()
        for values in logits.values():
            self.assertIsNotNone(values.grad)
            self.assertTrue(torch.isfinite(values.grad).all())
            self.assertLess(values.grad[0, 0].item(), 0)

    def test_batch_mean_matches_explicit_joint(self):
        torch.manual_seed(29)
        compatibility = torch.rand(3, 2, 4) > 0.5
        logits = {task: torch.randn(5, count) for task, count in zip(TASKS, (3, 2, 4))}
        labels = torch.zeros(5, 3, dtype=torch.long)
        result = HierarchicalLoss(compatibility)(logits, labels)
        probabilities = [logits[task].softmax(-1) for task in TASKS]
        joint = probabilities[0][:, :, None, None] * probabilities[1][:, None, :, None] * probabilities[2][:, None, None, :]
        expected = (joint * ~compatibility).sum((1, 2, 3)).mean()
        self.assertTrue(torch.allclose(result["lvp"], expected))

    def test_zero_penalty_all_valid_and_optional_graph(self):
        logits = {task: torch.randn(3, 2, requires_grad=True) for task in TASKS}
        labels = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 1]])
        result = HierarchicalLoss(torch.ones(2, 2, 2, dtype=torch.bool))(logits, labels)
        self.assertEqual(result["lvp"].item(), 0)
        flat = HierarchicalLoss(None, 0)(logits, labels)
        expected = sum(F.cross_entropy(logits[task], labels[:, index]) for index, task in enumerate(TASKS))
        self.assertTrue(torch.allclose(flat["total"], expected))
        with self.assertRaisesRegex(ValueError, "explicit compatibility"):
            HierarchicalLoss(None, 5)

    def test_reject_invalid_shapes_and_labels(self):
        loss = HierarchicalLoss(torch.ones(2, 2, 2, dtype=torch.bool))
        logits = {task: torch.zeros(1, 2) for task in TASKS}
        with self.assertRaisesRegex(ValueError, "outside"):
            loss(logits, torch.tensor([[2, 0, 0]]))
        with self.assertRaisesRegex(ValueError, "integer"):
            loss(logits, torch.zeros(1, 3))
        with self.assertRaisesRegex(ValueError, "exactly 0/1"):
            HierarchicalLoss(torch.full((2, 2, 2), 0.2))


class CompatibilityTests(unittest.TestCase):
    def load(self, document, num_classes=(2, 2, 2)):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compatibility.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_compatibility(path, num_classes)

    def test_triplets_and_dense_have_same_meaning(self):
        expected = torch.zeros(2, 2, 2, dtype=torch.bool)
        expected[1, 0, 1] = True
        self.assertTrue(torch.equal(self.load({"valid_triplets": [[1, 0, 1]]}), expected))
        self.assertTrue(torch.equal(self.load({"compatibility": expected.tolist()}), expected))

    def test_invalid_graph_input(self):
        bad = [
            {},
            {"valid_triplets": [[0, 0, 2]]},
            {"valid_triplets": [[True, 0, 1]]},
            {"valid_triplets": [[0.0, 0, 1]]},
            {"compatibility": [[[0.5] * 2] * 2] * 2},
            {"compatibility": [[[1] * 2] * 2]},
            {"compatibility": [[[1] * 2] * 2] * 2, "valid_triplets": []},
        ]
        for document in bad:
            with self.subTest(document=document), self.assertRaises(ValueError):
                self.load(document)


class MetricTests(unittest.TestCase):
    def test_ties_use_grouped_thresholds(self):
        # All scores tie: each binary AP is its prevalence (1/2), not 1.
        probabilities = {task: torch.full((4, 2), 0.5) for task in TASKS}
        labels = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]])
        result = evaluate_predictions(probabilities, labels)
        for task in TASKS:
            self.assertEqual(result[f"{task}_mAP"], 0.5)
            self.assertEqual(result[f"{task}_accuracy"], 0.5)
        self.assertNotIn("consistency", result)
        self.assertEqual(result, evaluate_predictions({task: values.flip(0) for task, values in probabilities.items()}, labels.flip(0)))
        json.dumps(result)

    def test_absent_class_zero_and_coverage(self):
        probabilities = {task: np.array([[0.9, 0.1], [0.8, 0.2]]) for task in TASKS}
        result = evaluate_predictions(probabilities, np.zeros((2, 3), dtype=int))
        self.assertEqual(result["action_mAP"], 0.5)
        self.assertEqual(result["action_classes_with_positives"], 1)
        self.assertEqual(result["action_num_classes"], 2)
        self.assertEqual(result["action_accuracy"], 1)

    def test_raw_argmax_consistency(self):
        probabilities = {task: np.array([[0.9, 0.1], [0.2, 0.8]]) for task in TASKS}
        compatibility = torch.zeros(2, 2, 2, dtype=torch.bool)
        compatibility[0, 0, 0] = True
        labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
        result = evaluate_predictions(probabilities, labels, compatibility)
        self.assertEqual(result["consistency"], 0.5)
        self.assertEqual(result["mean_mAP"], 1)

    def test_nontrivial_ap_and_rejection_of_logits(self):
        # Positive ordering is [1,0,1,0]: AP=(1+2/3)/2 for both classes.
        values = np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
        labels = np.repeat(np.array([[0], [1], [0], [1]]), 3, axis=1)
        probabilities = {task: values.copy() for task in TASKS}
        result = evaluate_predictions(probabilities, labels)
        self.assertAlmostEqual(result["mean_mAP"], 5 / 6)
        probabilities["action"] = np.ones((4, 2))
        with self.assertRaisesRegex(ValueError, "summing to 1"):
            evaluate_predictions(probabilities, labels)


if __name__ == "__main__":
    unittest.main()
