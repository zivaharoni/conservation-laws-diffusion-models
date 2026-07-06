from __future__ import annotations

import unittest
import torch

from src.exit import (
    apply_sequence_masking,
    iid_exact_posteriors,
    integrate_exit_curve,
    markov_masked_posteriors,
    markov_masked_posteriors_batch,
)
from src.models import BidirectionalMaskedTokenModel


class ExitUtilitiesTest(unittest.TestCase):
    def test_sequence_masking_forces_one_mask_per_empty_row(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.long)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        masked_tokens, mask, forced_count = apply_sequence_masking(
            tokens,
            torch.zeros(tokens.shape[0], dtype=torch.float32),
            mask_token_id=9,
            generator=generator,
        )
        self.assertEqual(forced_count, tokens.shape[0])
        self.assertTrue(torch.equal(mask.sum(dim=1), torch.ones(tokens.shape[0], dtype=torch.long)))
        self.assertTrue(torch.all(masked_tokens[mask] == 9))

    def test_sequence_masking_respects_extreme_probabilities(self) -> None:
        tokens = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(11)
        _, mask, _ = apply_sequence_masking(
            tokens,
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            mask_token_id=3,
            generator=generator,
        )
        self.assertEqual(int(mask[0].sum().item()), 1)
        self.assertEqual(int(mask[1].sum().item()), tokens.shape[1])

    def test_bidirectional_model_forward_shape(self) -> None:
        model = BidirectionalMaskedTokenModel(
            alphabet_size=5,
            embed_dim=12,
            n_transformer_layers=2,
            n_heads=3,
            attention_dropout=0.0,
            max_seq_len=8,
        )
        tokens = torch.tensor([[0, 1, 5, 2], [3, 4, 0, 5]], dtype=torch.long)
        logits = model(tokens)
        self.assertEqual(model.input_alphabet_size, 6)
        self.assertEqual(model.mask_token_id, 5)
        self.assertEqual(tuple(logits.shape), (2, 4, 5))

    def test_bidirectional_model_accepts_optional_channel_tilt(self) -> None:
        model = BidirectionalMaskedTokenModel(
            alphabet_size=4,
            embed_dim=8,
            n_transformer_layers=2,
            n_heads=2,
            attention_dropout=0.0,
            max_seq_len=8,
        )
        tokens = torch.tensor([[0, 1, 4, 2]], dtype=torch.long)
        outputs = model.forward_outputs(tokens)
        tilt = torch.full_like(outputs.logits_ext, 0.25)
        tilted = model.forward_outputs(tokens, channel_tilt=tilt)
        self.assertTrue(torch.allclose(tilted.logits, tilted.logits_ext + tilt))
        self.assertTrue(torch.allclose(outputs.logits, outputs.logits_ext))

    def test_integrate_exit_curve_matches_trapezoid_rule(self) -> None:
        s_grid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        exit_curve = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        self.assertAlmostEqual(integrate_exit_curve(s_grid, exit_curve), 1.0, places=8)

    def test_iid_exact_posteriors_masked_and_unmasked(self) -> None:
        probs = torch.tensor([0.25, 0.75], dtype=torch.float64)
        observed = torch.tensor([1, 0], dtype=torch.long)
        mask = torch.tensor([True, False], dtype=torch.bool)
        posteriors = iid_exact_posteriors(observed, mask, probs)
        self.assertTrue(torch.allclose(posteriors[0], probs, atol=1e-10))
        self.assertTrue(torch.allclose(posteriors[1], torch.tensor([1.0, 0.0], dtype=torch.float64), atol=1e-10))

    def test_markov_masked_posteriors_normalize(self) -> None:
        init_probs = torch.tensor([0.5, 0.5], dtype=torch.float64)
        transition_matrix = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64)
        observed = torch.tensor([0, 0, 1], dtype=torch.long)
        mask = torch.tensor([False, True, False], dtype=torch.bool)
        posteriors = markov_masked_posteriors(observed, mask, init_probs, transition_matrix)
        self.assertTrue(torch.allclose(posteriors.sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-10))
        self.assertTrue(torch.allclose(posteriors[0], torch.tensor([1.0, 0.0], dtype=torch.float64), atol=1e-10))
        self.assertTrue(torch.allclose(posteriors[2], torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-10))

    def test_permuted_markov_posteriors_match_chain_order_reference(self) -> None:
        init_probs = torch.tensor([0.6, 0.4], dtype=torch.float64)
        transition_matrix = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
        observed_chain = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        mask_chain = torch.tensor([False, True, True, False], dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        observed_perm = observed_chain[permutation]
        mask_perm = mask_chain[permutation]

        posterior_chain = markov_masked_posteriors(observed_chain, mask_chain, init_probs, transition_matrix)
        posterior_perm = markov_masked_posteriors_batch(
            observed_perm.unsqueeze(0),
            mask_perm.unsqueeze(0),
            init_probs,
            transition_matrix,
            sequence_permutation=permutation,
        )[0]
        self.assertTrue(torch.allclose(posterior_perm, posterior_chain[permutation], atol=1e-10))


if __name__ == "__main__":
    unittest.main()
