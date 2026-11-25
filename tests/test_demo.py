import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from hintnet.data import UAVHintDataset, validate_manifest_splits


class DemoTests(unittest.TestCase):
    def test_demo_has_all_classes_disjoint_videos_and_reproducible_features(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "make_demo.py"
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "one", Path(temporary) / "two"
            for directory in (first, second):
                subprocess.run([sys.executable, str(script), "--output", str(directory), "--feature-dim", "32"],
                               check=True, capture_output=True, text=True)
            counts = validate_manifest_splits(first / "manifest.jsonl")
            self.assertEqual(counts, {"train": 56, "val": 56, "test": 56})
            config = json.loads((first / "smoke.json").read_text())
            summary = json.loads((first / "demo_summary.json").read_text())
            self.assertTrue(summary["synthetic"])
            self.assertTrue(config["model"]["random_decoder"])
            for split in ("train", "val", "test"):
                dataset = UAVHintDataset(first / "manifest.jsonl", split=split, input_mode="tokens", token_dim=32)
                samples = [dataset[index] for index in range(len(dataset))]
                targets = torch.stack([sample["labels"] for sample in samples])
                self.assertEqual([len(targets[:, i].unique()) for i in range(3)], [7, 8, 4])
                self.assertGreater(len({sample["tokens"].shape[0] for sample in samples}), 1)
            for directory in ("train", "val", "test"):
                path = Path("tokens") / directory / "000.pt"
                self.assertTrue(torch.equal(torch.load(first / path, weights_only=True),
                                            torch.load(second / path, weights_only=True)))


if __name__ == "__main__":
    unittest.main()
