import unittest

import torch

from hintnet.model import (HINTNet, HierarchicalQueryRefinement, ModelConfig,
                           progressive_attention_mask)


def tiny_config(**overrides):
    args = dict(feature_dim=32, num_queries=2, num_prefix=2,
                cross_attention_heads=4, dropout=0.0, random_decoder=True,
                decoder_dim=48, decoder_layers=6, decoder_heads=4,
                decoder_ff_dim=96)
    args.update(overrides)
    return ModelConfig(**args)


class HINTNetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_hqr_equations_and_broadcast_gates(self):
        hqr = HierarchicalQueryRefinement(2, 4)
        with torch.no_grad():
            hqr.queries[0].fill_(1)
            hqr.queries[1].fill_(3)
            hqr.queries[2].fill_(8)
        result = hqr()
        torch.testing.assert_close(result[0], torch.ones(2, 4))
        torch.testing.assert_close(result[1], torch.full((2, 4), 2.0))
        torch.testing.assert_close(result[2], torch.full((2, 4), 5.0))

    def test_ptu_boundaries_mask_keys_not_queries(self):
        for index in range(12):
            mask = progressive_attention_mask(index, 12, 16, 16)
            visible = (32, 48, 64)[index // 4]
            self.assertTrue((mask[..., :visible] == 0).all())
            self.assertTrue(torch.isneginf(mask[..., visible:]).all())
            # Last query row is active even before its token is visible as a key.
            self.assertEqual(mask[0, 0, -1, 0].item(), 0.0)

    def test_actual_decoder_attentions_unveil_each_stage(self):
        model = HINTNet(tiny_config())
        embedded = torch.randn(2, 8, 48, requires_grad=True)
        result, attentions = model.decoder(embedded, return_attentions=True)
        for layer, attention in enumerate(attentions):
            visible = (4, 6, 8)[layer // 2]
            self.assertEqual(attention[..., visible:].abs().sum().item(), 0.0)
            torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
            self.assertGreater(attention[..., visible - 1].sum().item(), 0.0)
        result.square().mean().backward()
        self.assertGreater(embedded.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))

    def test_frozen_decoder_still_trains_upstream_parameters(self):
        model = HINTNet(tiny_config()).train()
        self.assertFalse(model.decoder.training)
        self.assertFalse(model.decoder.stack.training)
        logits = model(tokens=torch.randn(2, 5, 32))
        loss = sum(torch.nn.functional.cross_entropy(value, torch.tensor([0, 1]))
                   for value in logits.values())
        loss.backward()
        for name, parameter in [("prefix", model.prefix), ("queries", model.hqr.queries),
                                ("short_gate", model.hqr.short_gate),
                                ("long_gate", model.hqr.long_gate),
                                ("projection", model.projection[0].weight)]:
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertTrue(all(not p.requires_grad for p in model.decoder.parameters()))

    def test_masked_input_tokens_do_not_change_predictions(self):
        model = HINTNet(tiny_config()).eval()
        tokens = torch.randn(2, 5, 32)
        mask = torch.tensor([[True, True, False, False, False]]).expand(2, -1)
        before = model(tokens=tokens, token_mask=mask)
        tokens[:, 2:] += 100
        after = model(tokens=tokens, token_mask=mask)
        for name in before:
            torch.testing.assert_close(before[name], after[name])
        with self.assertRaises(ValueError):
            model(tokens=tokens, token_mask=torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
