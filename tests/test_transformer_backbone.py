from __future__ import annotations

import unittest

import torch

from src.transformer_backbone import GPT2StyleBlock


class TransformerBackboneAttentionMaskTest(unittest.TestCase):
    def _make_uniform_attention_block(self) -> GPT2StyleBlock:
        block = GPT2StyleBlock(hidden_size=4, n_heads=1, ffn_hidden_size=8, dropout=0.0, cond_dim=None)
        with torch.no_grad():
            block.qkv.weight.zero_()
            block.qkv.bias.zero_()
            block.qkv.weight[8:12].copy_(torch.eye(4))
            block.attn_out.weight.copy_(torch.eye(4))
            block.attn_out.bias.zero_()
        return block

    def test_bidirectional_attention_masks_self_token(self) -> None:
        block = self._make_uniform_attention_block()
        x = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        attn = block._self_attention(x, is_causal=False)
        expected = torch.tensor(
            [[[4.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(attn, expected, atol=1e-6))

    def test_bidirectional_attention_handles_single_token_sequence(self) -> None:
        block = self._make_uniform_attention_block()
        x = torch.tensor([[[2.0, -1.0, 0.5, 3.0]]], dtype=torch.float32)
        attn = block._self_attention(x, is_causal=False)
        self.assertTrue(torch.allclose(attn, torch.zeros_like(x), atol=1e-6))
        self.assertTrue(torch.isfinite(attn).all())

    def test_causal_attention_keeps_current_token_visible(self) -> None:
        block = self._make_uniform_attention_block()
        x = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        attn = block._self_attention(x, is_causal=True)
        expected = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(attn, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
