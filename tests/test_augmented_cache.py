import json
from pathlib import Path
import tempfile
import unittest

import torch

from hintnet.data import UAVHintDataset, sample_frame_indices


class AugmentedCacheTests(unittest.TestCase):
    def make_dataset(self, folder, training, frames=8, malformed=False):
        original = torch.full((3, 32), 2.0)
        flipped = torch.full((3, 32), 9.0)
        if malformed:
            flipped = flipped[:2]
        torch.save({"tokens": original, "flipped_tokens": flipped,
                    "token_mask": torch.tensor([True, True, False])}, folder / "tokens.pt")
        row = {"video_id": "001", "instance_id": "person", "split": "train",
               "frames": [f"unused_{index}.jpg" for index in range(frames)],
               "tokens": "tokens.pt", "labels": [0, 0, 0]}
        (folder / "manifest.jsonl").write_text(json.dumps(row) + "\n")
        return UAVHintDataset(folder / "manifest.jsonl", training=training,
                              input_mode="tokens", token_dim=32, num_frames=8)

    def test_training_variants_match_raw_flip_bit_and_rng_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary), True)
            observed = set()
            for seed in range(8):
                torch.manual_seed(seed)
                flip = bool(torch.rand(()) < 0.5)
                sample_frame_indices(8, 8, training=True)
                expected_rng = torch.get_rng_state()
                torch.manual_seed(seed)
                sample = dataset[0]
                self.assertEqual(sample["tokens"][0, 0].item(), 9.0 if flip else 2.0)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertEqual(sample["token_mask"].tolist(), [True, True, False])
                observed.add(flip)
            self.assertEqual(observed, {True, False})

    def test_eval_always_uses_original_without_rng_draws(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary), False)
            torch.manual_seed(17)
            before = torch.get_rng_state()
            sample = dataset[0]
            self.assertEqual(sample["tokens"][0, 0].item(), 2.0)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_variable_temporal_sampling_and_bad_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dataset = self.make_dataset(folder, True, frames=10)
            with self.assertRaisesRegex(ValueError, "exactly num_frames"):
                dataset[0]
            dataset = self.make_dataset(folder, True, malformed=True)
            with self.assertRaisesRegex(ValueError, "same shape"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
