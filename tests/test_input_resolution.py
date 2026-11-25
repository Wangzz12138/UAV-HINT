"""Resolution transfer checks without downloads, GPU use, or new learned weights."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from hintnet.backbones import FrozenActionCLIP
from hintnet.data import UAVHintDataset, collate_batch
from hintnet.vendor.actionclip.model import VisualTransformer
from tools.extract_augmented_features import (FlippedObservationDataset,
                                               encoder_inputs, validate_cache)


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisualTransformer(224, 16, 32, 1, 4, 512)

    def encode_image(self, value):
        return self.visual(value)


class TinyTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_position_embeddings = nn.Embedding(8, 512)

    def forward(self, value):
        return value.mean(dim=1)


def plain_pretrained_grid_forward(model, images):
    """Unchanged pre-transfer VisualTransformer.forward for joint=False."""
    x = model.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat((cls, x), dim=1) + model.positional_embedding.to(x.dtype)
    if model.emb_dropout > 0:
        x = model.dropout(x)
    x = model.ln_pre(x)
    x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = model.ln_post(x[:, 0, :])
    return x @ model.proj if model.proj is not None else x


class InputResolutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def checkpoint(self, folder):
        path = Path(folder) / "tiny_actionclip.pt"
        torch.save({"model_state_dict": {
            "visual.proj": torch.zeros(32, 512),
            "visual.conv1.weight": torch.zeros(32, 3, 16, 16),
            "text_projection": torch.zeros(32, 512)},
            "fusion_model_state_dict": {"frame_position_embeddings.weight": torch.zeros(8, 512)}}, path)
        return path

    def encoder(self, checkpoint, resolution):
        with patch("hintnet.vendor.actionclip.model.build_model", return_value=TinyClip()), \
             patch("hintnet.vendor.actionclip.visual_prompt.visual_prompt", return_value=TinyTemporal()):
            return FrozenActionCLIP(checkpoint, num_frames=2, input_resolution=resolution,
                                    use_text=False, image_batch_size=2)

    def test_original_224_forward_and_parameter_storage_remain_bitwise_identical(self):
        model = VisualTransformer(224, 16, 32, 2, 4, 24).eval()
        images = torch.randn(2, 3, 224, 224)
        before = deepcopy(model.state_dict())
        parameter_object = model.positional_embedding
        with torch.no_grad():
            expected = plain_pretrained_grid_forward(model, images)
            actual = model(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIs(model.positional_embedding, parameter_object)
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_bicubic_spatial_positions_keep_cls_and_pretrained_parameter_unchanged(self):
        model = VisualTransformer(224, 16, 32, 1, 4, 24)
        before = model.positional_embedding.detach().clone()
        parameter_count = sum(p.numel() for p in model.parameters())
        spatial = before[1:].reshape(14, 14, 32).permute(2, 0, 1).unsqueeze(0)
        for side in (18, 21):
            expected = F.interpolate(spatial, size=(side, side), mode="bicubic", align_corners=False)
            expected = expected.squeeze(0).permute(1, 2, 0).reshape(side * side, 32)
            actual = model.positional_embedding_for_grid(side, side)
            self.assertEqual(tuple(actual.shape), (side * side + 1, 32))
            torch.testing.assert_close(actual[:1], before[:1], rtol=0, atol=0)
            torch.testing.assert_close(actual[1:], expected, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
        self.assertEqual(tuple(model.positional_embedding.shape), (197, 32))
        torch.testing.assert_close(model.positional_embedding, before, rtol=0, atol=0)

    def test_wrapper_accepts_true_288_and_336_inputs_without_unfreezing_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.checkpoint(temporary)
            for size, grid in ((288, 18), (336, 21)):
                model = self.encoder(checkpoint, size)
                before = model.clip.visual.positional_embedding.detach().clone()
                observed = []
                handle = model.clip.visual.conv1.register_forward_hook(lambda module, inputs, output: observed.append(tuple(output.shape[-2:])))
                video = torch.randn(1, 2, 3, size, size)
                tokens, mask = model(video=video, local_video=video)
                handle.remove()
                self.assertEqual(tuple(tokens.shape), (1, 2, 512))
                self.assertEqual(observed, [(grid, grid), (grid, grid)])
                self.assertTrue(mask.all() and torch.isfinite(tokens).all())
                self.assertFalse(tokens.requires_grad)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                self.assertEqual(model.clip.visual.input_resolution, 224)
                self.assertEqual(model.input_resolution, size)
                torch.testing.assert_close(model.clip.visual.positional_embedding, before, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "Configured ActionCLIP"):
                    model(video=video[:, :, :, :224, :224], local_video=video)
                model.train(True)
                self.assertFalse(model.training)

    def test_resolution_validation_rejects_non_patch_multiple_before_model_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.checkpoint(temporary)
            for value in (0, True, 288.0):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    FrozenActionCLIP(checkpoint, input_resolution=value)
            with self.assertRaisesRegex(ValueError, "divisible"):
                FrozenActionCLIP(checkpoint, input_resolution=225)

    def test_resolution_changes_neither_parameter_initialization_nor_rng_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.checkpoint(temporary)
            reference_state = reference_rng = None
            for size in (None, 224, 288, 336):
                torch.manual_seed(29)
                model = self.encoder(checkpoint, size)
                state = model.state_dict()
                rng = torch.get_rng_state()
                if reference_state is None:
                    reference_state = deepcopy(state)
                    reference_rng = rng.clone()
                else:
                    self.assertEqual(list(state), list(reference_state))
                    for name, value in state.items():
                        torch.testing.assert_close(value, reference_state[name], rtol=0, atol=0)
                    self.assertTrue(torch.equal(rng, reference_rng))

    def test_high_resolution_original_and_flip_cache_match_actual_raw_training_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            y, x = np.mgrid[:72, :96]
            image = np.stack((x * 2, y * 3, (x + y) % 255), axis=-1).astype(np.uint8)
            Image.fromarray(image).save(folder / "frame.png")
            row = {"video_id": "001", "instance_id": "person", "split": "train",
                   "frames": ["frame.png", "frame.png"], "bboxes": [[7, 5, 80, 69]] * 2,
                   "text": {"scene": "street"}, "labels": [0, 0, 0]}
            manifest = folder / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            options = dict(num_frames=2, image_size=336)
            original = UAVHintDataset(manifest, training=False, **options)
            flipped = FlippedObservationDataset(manifest, training=False, **options)
            raw_training = UAVHintDataset(manifest, training=True, **options)
            model = self.encoder(self.checkpoint(folder), 336)
            ordinary, mask = model(**encoder_inputs(collate_batch([original[0]]), "cpu"))
            mirrored, _ = model(**encoder_inputs(collate_batch([flipped[0]]), "cpu"))
            self.assertGreater((ordinary - mirrored).abs().max().item(), 0)
            path = folder / "features.pt"
            torch.save({"tokens": ordinary[0], "flipped_tokens": mirrored[0], "token_mask": mask[0]}, path)
            checked = validate_cache(model, raw_training, original, flipped, [path], [0], "cpu", 1e-6)
            self.assertEqual(checked["status"], "passed")
            self.assertEqual(checked["maximum_token_absolute_difference"], 0.)


if __name__ == "__main__":
    unittest.main()
