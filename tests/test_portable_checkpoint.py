"""Portable state restoration must be complete and retain the exact frozen base."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hintnet.checkpoints import (ADAPTER_FORMAT, frozen_decoder_state,
                                load_adapter_state, tensor_state_sha256,
                                trainable_state)
from hintnet.runtime import DEFAULT_CONFIG, load_saved_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(3, 3).requires_grad_(False)
        self.prefix = nn.Parameter(torch.zeros(1, 3))
        self.head = nn.Linear(3, 2)

    def forward(self, value):
        return self.head(self.decoder(value + self.prefix))


class PortableCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.source = TinyModel()
        torch.manual_seed(43)
        self.target = TinyModel()
        self.target.decoder.load_state_dict(self.source.decoder.state_dict())
        self.payload = {"format": ADAPTER_FORMAT, "format_version": 1,
                        "model": trainable_state(self.source),
                        "source": {"frozen_decoder_sha256": tensor_state_sha256(frozen_decoder_state(self.source))}}

    def test_complete_state_restoration_reproduces_outputs_and_preserves_frozen_base(self):
        before = {name: value.clone() for name, value in frozen_decoder_state(self.target).items()}
        summary = load_adapter_state(self.target, self.payload)
        self.assertEqual(summary["trainable_parameters"], 11)
        value = torch.randn(4, 3)
        torch.testing.assert_close(self.source(value), self.target(value), rtol=0, atol=0)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, self.target.state_dict()[name]))

    def test_missing_and_unknown_tensors_are_rejected(self):
        missing = deepcopy(self.payload)
        missing["model"].pop("prefix")
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            load_adapter_state(self.target, missing)
        unknown = deepcopy(self.payload)
        unknown["model"]["decoder.weight"] = self.source.decoder.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            load_adapter_state(self.target, unknown)

    def test_shape_dtype_and_nonfinite_parameters_are_rejected(self):
        for corrupt in (torch.zeros(3, 3), torch.zeros(1, 3, dtype=torch.float64),
                        torch.full((1, 3), float("nan"))):
            payload = deepcopy(self.payload)
            payload["model"]["prefix"] = corrupt
            with self.assertRaises(ValueError):
                load_adapter_state(self.target, payload)

    def test_changed_frozen_decoder_is_rejected_before_trainable_state_changes(self):
        before = trainable_state(self.target)
        with torch.no_grad():
            self.target.decoder.bias.add_(.01)
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            load_adapter_state(self.target, self.payload)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, self.target.state_dict()[name]))

    def test_fingerprint_includes_names_shapes_dtypes_and_exact_values(self):
        a = torch.arange(4, dtype=torch.float32)
        original = tensor_state_sha256({"a": a})
        for changed in ({"b": a}, {"a": a.reshape(2, 2)}, {"a": a.double()}, {"a": a + 1}):
            self.assertNotEqual(original, tensor_state_sha256(changed))
        self.assertEqual(tensor_state_sha256({"a": a, "b": a + 1}),
                         tensor_state_sha256({"b": a + 1, "a": a}))

    def test_runtime_adapter_branch_uses_normal_pretrained_constructor(self):
        payload = deepcopy(self.payload)
        payload["config"] = deepcopy(DEFAULT_CONFIG)
        payload["config"]["data"]["input_mode"] = "tokens"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(payload, path)
            # An adapter is safe to deserialize with the tensor-only unpickler.
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["format"], ADAPTER_FORMAT)
            with patch("hintnet.runtime.make_model", return_value=self.target) as constructor:
                model, config, saved = load_saved_model(path, torch.device("cpu"))
            constructor.assert_called_once_with(config, torch.device("cpu"))
            self.assertIs(model, self.target)
            self.assertEqual(saved["adapter_validation"]["trainable_tensors"], 3)

    def test_raw_runtime_rejects_another_encoder_before_constructing_the_model(self):
        payload = deepcopy(self.payload)
        payload["config"] = deepcopy(DEFAULT_CONFIG)
        payload["source"]["actionclip_checkpoint_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            encoder = Path(directory) / "wrong_encoder.pt"
            encoder.write_bytes(b"another checkpoint")
            payload["config"]["backbone"]["checkpoint"] = str(encoder)
            checkpoint = Path(directory) / "adapter.pt"
            torch.save(payload, checkpoint)
            with patch("hintnet.runtime.make_model") as constructor:
                with self.assertRaisesRegex(ValueError, "ActionCLIP checkpoint fingerprint differs"):
                    load_saved_model(checkpoint, torch.device("cpu"))
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
