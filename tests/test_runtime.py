"""Runtime regressions: exact decoder restoration, LR endpoints and inference."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import T5Config, T5ForConditionalGeneration

from hintnet.model import ModelConfig
from hintnet.runtime import (
    DEFAULT_CONFIG,
    atomic_save,
    evaluate,
    learning_rate_factor,
    load_saved_model,
    make_model,
    merge_config,
)


class FixedPredictions(torch.nn.Module):
    def forward(self, tokens, **kwargs):
        values = torch.tensor([[2.0, -1.0]], device=tokens.device).expand(len(tokens), -1)
        return {name: values for name in ("action", "short_term", "long_term")}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_schedule_regular_warmup_and_zero_warmup(self):
        factors = [learning_rate_factor(step, 10, 2) for step in range(10)]
        self.assertEqual(factors[:3], [0.5, 1.0, 1.0])
        self.assertEqual(factors[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(factors[2:], factors[3:])))
        self.assertEqual(learning_rate_factor(0, 10, 0), 1.0)
        self.assertEqual(learning_rate_factor(9, 10, 0), 0.0)
        self.assertEqual(learning_rate_factor(0, 1, 0), 1.0)

    def test_schedule_reaches_zero_with_one_decay_update(self):
        self.assertEqual(learning_rate_factor(1, 2, 1), 0.0)
        self.assertEqual(learning_rate_factor(2, 3, 2), 0.0)

    def test_pretrained_decoder_restores_without_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "pretrained"
            native_config = T5Config(
                vocab_size=32, d_model=48, d_ff=96, num_heads=4, d_kv=12,
                num_layers=6, num_decoder_layers=6, dropout_rate=0.0,
            )
            T5ForConditionalGeneration(native_config).save_pretrained(source)
            model_config = ModelConfig(
                feature_dim=32, num_queries=2, num_prefix=2,
                cross_attention_heads=4, dropout=0.0, t5_model=str(source),
                local_files_only=True, random_decoder=False,
            )
            config = merge_config(DEFAULT_CONFIG, {
                "model": model_config.to_dict(), "data": {"input_mode": "tokens"},
            })
            original = make_model(config, torch.device("cpu")).eval()
            tokens = torch.randn(2, 5, 32)
            expected = original(tokens=tokens)
            checkpoint = path / "hintnet.pt"
            atomic_save({
                "format_version": 1, "config": config,
                "decoder_config": original.decoder.stack.config.to_dict(),
                "model": original.state_dict(),
            }, checkpoint)
            shutil.rmtree(source)
            restored, _, _ = load_saved_model(checkpoint, torch.device("cpu"))
            restored.eval()
            actual = restored(tokens=tokens)
            for task in expected:
                torch.testing.assert_close(actual[task], expected[task], rtol=0, atol=0)
            self.assertTrue(all(not parameter.requires_grad for parameter in restored.decoder.parameters()))

    def make_batch(self, labels):
        return {
            "tokens": torch.zeros(len(labels), 3, 32),
            "labels": torch.tensor(labels),
            "video_id": [f"v{index}" for index in range(len(labels))],
            "instance_id": ["person"] * len(labels),
        }

    def test_unlabeled_prediction_omits_metrics_and_unknown_labels(self):
        batch = self.make_batch([[-1, -1, -1], [-1, -1, -1]])
        metrics, rows = evaluate(FixedPredictions(), [batch], torch.device("cpu"), collect_predictions=True)
        self.assertEqual(metrics, {})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("labels" not in row for row in rows))

    def test_mixed_prediction_evaluates_only_labeled_rows(self):
        batch = self.make_batch([[0, 0, 0], [-1, -1, -1]])
        metrics, rows = evaluate(FixedPredictions(), [batch], torch.device("cpu"), collect_predictions=True)
        self.assertEqual(metrics["num_samples"], 1)
        self.assertEqual(metrics["action_accuracy"], 1.0)
        self.assertEqual(rows[0]["labels"], [0, 0, 0])
        self.assertNotIn("labels", rows[1])


if __name__ == "__main__":
    unittest.main()
