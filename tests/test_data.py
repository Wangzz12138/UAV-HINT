import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from hintnet.data import (
    CLIP_MEAN, CLIP_STD, UAVHintDataset, collate_batch, encode_labels,
    format_text, sample_frame_indices, validate_manifest_splits,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def manifest(self, records):
        path = self.root / "manifest.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def record(self, **extra):
        return {"video_id": "v1", "instance_id": "person1", "split": "train",
                "labels": {"action": "Sitting", "short_term": "Keeping", "long_term": "Daily routine"}, **extra}

    def test_video_split_leakage_rejected_before_selecting_split(self):
        manifest = self.manifest([self.record(tokens="a.pt"), self.record(split="test", tokens="b.pt")])
        with self.assertRaisesRegex(ValueError, "video leakage"):
            UAVHintDataset(manifest, split="train", input_mode="tokens")
        with self.assertRaisesRegex(ValueError, "video leakage"):
            validate_manifest_splits(manifest)

    def test_labels_are_explicit_and_context_does_not_encode_targets(self):
        self.assertEqual(encode_labels({"action": "Standing", "short_term": "Waiting", "long_term": "Undetermined"}).tolist(), [6, 7, 3])
        for labels in ([7, 0, 0], [True, 0, 0], [0, 0], [0, 0, 0.0]):
            with self.assertRaises(ValueError):
                encode_labels(labels)
        self.assertEqual(format_text({"scene": "park", "action": "Running", "long_term": "Daily routine"}), "scene: park")

    def test_local_bbox_and_deterministic_uniform_sampling(self):
        paths, boxes = [], []
        for i in range(4):
            pixels = np.zeros((8, 12, 3), dtype=np.uint8)
            pixels[:, :6] = (255, 0, 0)
            pixels[:, 6:] = (0, 0, 255)
            filename = f"{i}.png"
            Image.fromarray(pixels).save(self.root / filename)
            paths.append(filename)
            boxes.append([0, 0, 6, 8])
        manifest = self.manifest([self.record(frames=paths, bboxes=boxes)])
        dataset = UAVHintDataset(manifest, num_frames=3, image_size=8)
        first, second = dataset[0], dataset[0]
        self.assertEqual(tuple(first["video"].shape), (3, 3, 8, 8))
        self.assertTrue(torch.equal(first["video"], second["video"]))
        restored = first["local_video"] * torch.tensor(CLIP_STD).view(1, 3, 1, 1) + torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(restored[:, 0], torch.ones(3, 8, 8), atol=1e-6))
        self.assertTrue(torch.allclose(restored[:, 1:], torch.zeros(3, 2, 8, 8), atol=1e-6))
        self.assertEqual(sample_frame_indices(4, 3), [0, 2, 3])
        self.assertEqual(sample_frame_indices(1, 3), [0, 0, 0])

    def test_training_crop_and_full_frame_share_augmentation(self):
        pixels = np.zeros((8, 12, 3), dtype=np.uint8)
        pixels[:, :6] = (255, 0, 0)
        Image.fromarray(pixels).save(self.root / "frame.png")
        manifest = self.manifest([self.record(frames=["frame.png"] * 4, bboxes=[[0, 0, 12, 8]] * 4)])
        dataset = UAVHintDataset(manifest, num_frames=3, image_size=8, training=True)
        for _ in range(5):
            sample = dataset[0]
            self.assertTrue(torch.equal(sample["video"], sample["local_video"]))

    def test_token_cache_padding_and_masks(self):
        torch.save({"tokens": torch.ones(2, 512), "token_mask": torch.tensor([True, False])}, self.root / "one.pt")
        torch.save({"global": torch.ones(1, 512), "local": torch.ones(1, 512) * 2, "text": torch.ones(1, 512) * 3}, self.root / "two.pt")
        manifest = self.manifest([self.record(tokens="one.pt"), self.record(video_id="v2", tokens="two.pt")])
        dataset = UAVHintDataset(manifest, input_mode="tokens")
        batch = collate_batch([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["tokens"].shape), (2, 3, 512))
        self.assertEqual(batch["token_mask"].tolist(), [[True, False, False], [True, True, True]])
        self.assertEqual(batch["tokens"][1, :, 0].tolist(), [1, 2, 3])

    def test_bad_bbox_and_nonfinite_tokens_fail(self):
        manifest = self.manifest([self.record(frames=["a.png"], bboxes=[[1, 0, 0, 8]])])
        with self.assertRaisesRegex(ValueError, "bbox"):
            UAVHintDataset(manifest)
        torch.save(torch.full((2, 512), float("nan")), self.root / "bad.pt")
        dataset = UAVHintDataset(self.manifest([self.record(tokens="bad.pt")]), input_mode="tokens")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dataset[0]

    def test_manifest_tool_preserves_paths_when_moving_output(self):
        torch.save(torch.ones(2, 512), self.root / "tokens.pt")
        record = self.record(tokens="tokens.pt")
        record.pop("split")
        source = self.manifest([record])
        split_file = self.root / "splits.json"
        split_file.write_text(json.dumps({"train": ["v1"], "val": [], "test": []}))
        output = self.root / "nested" / "normalized.jsonl"
        script = Path(__file__).resolve().parents[1] / "tools" / "prepare_manifest.py"
        subprocess.run([sys.executable, str(script), str(source), "--output", str(output),
                        "--split-file", str(split_file), "--input-mode", "tokens", "--check-files"],
                       check=True, capture_output=True, text=True)
        dataset = UAVHintDataset(output, input_mode="tokens", split="train")
        self.assertEqual(tuple(dataset[0]["tokens"].shape), (2, 512))


if __name__ == "__main__":
    unittest.main()
