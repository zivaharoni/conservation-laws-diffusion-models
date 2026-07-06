from __future__ import annotations

import unittest

import torch

from src.channels import apply_ksc_noise, log_ksc_kernel
from src.img_models import RGB8ImageTokenUNet
from src.lm_objectives import compute_qsc_loss


class ImageQscTiltingTest(unittest.TestCase):
    def test_forward_outputs_applies_channel_tilt_when_conditioned(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(11)
        vocab_size = 256
        tokens = torch.randint(vocab_size, (2, 3072), generator=generator, dtype=torch.long)
        t = torch.tensor([0.25, 0.75], dtype=torch.float32)
        observed, _correct = apply_ksc_noise(tokens, t, vocab_size, generator=generator)
        kernel = log_ksc_kernel(observed, t, vocab_size)

        model = RGB8ImageTokenUNet(
            vocab_size=vocab_size,
            image_size=32,
            base_channels=32,
            channel_multipliers=(1, 2, 2, 2),
            dropout=0.0,
            time_embed_dim=128,
        )
        outputs = model.forward_outputs(observed, timesteps=t, channel_tilt=kernel)
        self.assertTrue(torch.allclose(outputs.logits, outputs.logits_ext + kernel, atol=1e-6, rtol=1e-6))

    def test_compute_qsc_loss_runs_with_unet(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(13)
        vocab_size = 256
        tokens = torch.randint(vocab_size, (2, 3072), generator=generator, dtype=torch.long)
        t = torch.tensor([0.25, 0.75], dtype=torch.float32)

        model = RGB8ImageTokenUNet(
            vocab_size=vocab_size,
            image_size=32,
            base_channels=32,
            channel_multipliers=(1, 2, 2, 2),
            dropout=0.0,
            time_embed_dim=128,
        )
        stats = compute_qsc_loss(model, tokens=tokens, t=t, generator=generator)
        self.assertTrue(torch.isfinite(stats["loss"]).all())


if __name__ == "__main__":
    unittest.main()
