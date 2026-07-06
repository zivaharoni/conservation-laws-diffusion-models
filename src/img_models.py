"""UNet image-token and pixel diffusion models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .channels import awgn
from .models import posterior_mean_from_logits
from .transformer_backbone import TransformerModelOutput, apply_channel_tilt


def _init_embedding_normal(embedding: nn.Embedding, mean: float = 0.0, std: float = 1.0) -> None:
    with torch.no_grad():
        nn.init.normal_(embedding.weight, mean=mean, std=std)


def _group_norm_groups(channels: int) -> int:
    for group_count in (32, 16, 8, 4, 2, 1):
        if channels % group_count == 0:
            return group_count
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(timesteps, dtype=torch.float32, device=timesteps.device)
        half_dim = self.embedding_dim // 2
        freq = torch.exp(
            -math.log(10_000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / max(half_dim - 1, 1)
        )
        args = t.unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DDPMResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_proj(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.activation(self.norm2(h))))
        return h + self.skip(x)


class DDPMSelfAttention2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_norm_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        h = self.norm(x).reshape(bsz, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        scale = channels ** -0.5
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.proj(out).reshape(bsz, channels, height, width)
        return x + out


class DDPMDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DDPMUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class CIFAR10DDPMUNet(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.1,
        time_embed_input_dim: int = 128,
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.channel_multipliers = tuple(int(value) for value in channel_multipliers)
        self.level_channels = [self.base_channels * multiplier for multiplier in self.channel_multipliers]
        self.time_embed_input_dim = int(time_embed_input_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.dropout = float(dropout)
        self.attention_resolutions = {16}
        self.num_res_blocks = 2

        self.input_conv = nn.Conv2d(self.in_channels, self.level_channels[0], kernel_size=3, padding=1)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_embed_input_dim),
            nn.Linear(self.time_embed_input_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = self.level_channels[0]
        current_resolution = self.image_size
        self.skip_channels = [current_channels]
        for level_idx, out_channels_level in enumerate(self.level_channels):
            level_blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                layers = nn.ModuleList(
                    [
                        DDPMResBlock(
                            current_channels,
                            out_channels_level,
                            time_embed_dim=self.time_embed_dim,
                            dropout=self.dropout,
                        ),
                        DDPMSelfAttention2d(out_channels_level)
                        if current_resolution in self.attention_resolutions
                        else nn.Identity(),
                    ]
                )
                level_blocks.append(layers)
                current_channels = out_channels_level
                self.skip_channels.append(current_channels)
            self.encoder_blocks.append(level_blocks)
            if level_idx < len(self.level_channels) - 1:
                self.downsamples.append(DDPMDownsample(current_channels))
                self.skip_channels.append(current_channels)
                current_resolution //= 2

        self.mid_block1 = DDPMResBlock(
            current_channels,
            current_channels,
            time_embed_dim=self.time_embed_dim,
            dropout=self.dropout,
        )
        self.mid_attn = DDPMSelfAttention2d(current_channels)
        self.mid_block2 = DDPMResBlock(
            current_channels,
            current_channels,
            time_embed_dim=self.time_embed_dim,
            dropout=self.dropout,
        )

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_levels = list(enumerate(self.level_channels))[::-1]
        for level_idx, out_channels_level in reversed_levels:
            current_resolution = self.image_size // (2**level_idx)
            level_blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                skip_channels = self.skip_channels.pop()
                layers = nn.ModuleList(
                    [
                        DDPMResBlock(
                            current_channels + skip_channels,
                            out_channels_level,
                            time_embed_dim=self.time_embed_dim,
                            dropout=self.dropout,
                        ),
                        DDPMSelfAttention2d(out_channels_level)
                        if current_resolution in self.attention_resolutions
                        else nn.Identity(),
                    ]
                )
                level_blocks.append(layers)
                current_channels = out_channels_level
            self.decoder_blocks.append(level_blocks)
            if level_idx > 0:
                self.upsamples.append(DDPMUpsample(current_channels))

        self.out_norm = nn.GroupNorm(_group_norm_groups(current_channels), current_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(current_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B, C, H, W], got {tuple(x.shape)}")
        if tuple(x.shape[1:]) != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(
                f"Expected x shape [B, {self.in_channels}, {self.image_size}, {self.image_size}], got {tuple(x.shape)}"
            )
        t = torch.as_tensor(timesteps, dtype=torch.float32, device=x.device)
        if t.ndim == 2 and int(t.shape[1]) == 1:
            t = t.squeeze(1)
        if t.ndim != 1 or int(t.shape[0]) != int(x.shape[0]):
            raise ValueError(f"Expected timesteps shape {(int(x.shape[0]),)}, got {tuple(t.shape)}")

        temb = self.time_embed(t)
        h = self.input_conv(x)
        skips: list[torch.Tensor] = []
        skips.append(h)

        for level_idx, level_blocks in enumerate(self.encoder_blocks):
            for block, attn in level_blocks:
                h = block(h, temb)
                h = attn(h)
                skips.append(h)
            if level_idx < len(self.downsamples):
                h = self.downsamples[level_idx](h)
                skips.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for level_idx, level_blocks in enumerate(self.decoder_blocks):
            for block, attn in level_blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, temb)
                h = attn(h)
            if level_idx < len(self.upsamples):
                h = self.upsamples[level_idx](h)

        if skips:
            raise RuntimeError(f"Expected all skip tensors to be consumed, found {len(skips)} remaining.")

        return self.out_conv(self.out_act(self.out_norm(h)))


def _rgb8_tokens_to_normalized_image(
    tokens: torch.Tensor,
    *,
    image_size: int,
    mask_token_id: int | None = None,
    mask_fill_value: float = 0.5,
) -> torch.Tensor:
    tokens_t = torch.as_tensor(tokens, dtype=torch.long, device=tokens.device)
    expected_seq_len = int(image_size) * int(image_size) * 3
    if tokens_t.ndim != 2 or int(tokens_t.shape[1]) != expected_seq_len:
        raise ValueError(f"Expected tokens [B, {expected_seq_len}], got {tuple(tokens_t.shape)}")
    safe_tokens = tokens_t.clone()
    mask = None
    if mask_token_id is not None:
        mask = safe_tokens == int(mask_token_id)
        safe_tokens = safe_tokens.clamp(min=0, max=255)
    image = safe_tokens.view(int(tokens_t.shape[0]), int(image_size), int(image_size), 3).to(dtype=torch.float32) / 255.0
    if mask is not None and bool(mask.any()):
        image[mask.view(int(tokens_t.shape[0]), int(image_size), int(image_size), 3)] = float(mask_fill_value)
    return image.permute(0, 3, 1, 2).contiguous().mul(2.0).sub(1.0)


def _normalized_image_to_rgb8_logits(image_logits: torch.Tensor) -> torch.Tensor:
    if image_logits.ndim != 4:
        raise ValueError(f"Expected image_logits [B, C, H, W], got {tuple(image_logits.shape)}")
    bsz, channels, height, width = image_logits.shape
    if channels % 3 != 0:
        raise ValueError(f"Expected channel count divisible by 3, got {channels}")
    vocab_size = channels // 3
    return image_logits.permute(0, 2, 3, 1).reshape(bsz, height * width * 3, vocab_size).contiguous()


def rgb8_normalized_values_to_tokens(values: torch.Tensor) -> torch.Tensor:
    return torch.round(torch.clamp((values + 32.0) * (255.0 / 64.0), min=0.0, max=255.0)).to(dtype=torch.long)


class RGB8ImageTokenUNet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 256,
        image_size: int = 32,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.1,
        time_embed_input_dim: int = 128,
        time_embed_dim: int = 512,
        input_skip: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.image_size = int(image_size)
        self.n_token_channels = 3
        self.max_seq_len = self.image_size * self.image_size * self.n_token_channels
        self.input_vocab_size = self.vocab_size + 1
        self.mask_token_id = self.vocab_size
        self.input_skip = bool(input_skip)
        self.backbone = CIFAR10DDPMUNet(
            image_size=self.image_size,
            in_channels=3,
            out_channels=3 * self.vocab_size,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            dropout=dropout,
            time_embed_input_dim=time_embed_input_dim,
            time_embed_dim=time_embed_dim,
        )
        if self.input_skip:
            self.input_skip_logits = nn.Embedding(self.vocab_size, self.vocab_size)
            self.input_skip_gate = nn.Linear(time_embed_dim, self.vocab_size)
            nn.init.zeros_(self.input_skip_logits.weight)
            nn.init.zeros_(self.input_skip_gate.weight)
            nn.init.zeros_(self.input_skip_gate.bias)

    def _resolve_timesteps(self, timesteps: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if timesteps is None:
            raise ValueError("timesteps must be provided for RGB8ImageTokenUNet.")
        t_t = torch.as_tensor(timesteps, dtype=torch.float32, device=device)
        if t_t.ndim == 2 and int(t_t.shape[1]) == 1:
            t_t = t_t.squeeze(1)
        if t_t.ndim != 1 or int(t_t.shape[0]) != batch_size:
            raise ValueError(f"Expected timesteps shape {(batch_size,)}, got {tuple(t_t.shape)}")
        return t_t

    def _input_skip_logits(self, tokens: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if not self.input_skip:
            return torch.zeros((*tokens.shape, self.vocab_size), dtype=torch.float32, device=tokens.device)
        real_token_mask = tokens != self.mask_token_id
        safe_tokens = tokens.clamp(min=0, max=self.vocab_size - 1)
        skip_logits = self.input_skip_logits(safe_tokens)
        skip_logits = skip_logits * real_token_mask.unsqueeze(-1).to(dtype=skip_logits.dtype)
        time_gate = 1.0 + self.input_skip_gate(self.backbone.time_embed(timesteps)).unsqueeze(1)
        return skip_logits * time_gate

    def forward_outputs(
        self,
        tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        del t_embed
        tokens_t = torch.as_tensor(tokens, dtype=torch.long, device=tokens.device)
        image = _rgb8_tokens_to_normalized_image(tokens_t, image_size=self.image_size, mask_token_id=self.mask_token_id)
        resolved_timesteps = self._resolve_timesteps(timesteps, batch_size=int(tokens_t.shape[0]), device=tokens_t.device)
        logits_ext = _normalized_image_to_rgb8_logits(self.backbone(image, resolved_timesteps))
        if self.input_skip:
            logits_ext = logits_ext + self._input_skip_logits(tokens_t, resolved_timesteps).to(dtype=logits_ext.dtype)
        return TransformerModelOutput(hidden=logits_ext, logits_ext=logits_ext, logits=apply_channel_tilt(logits_ext, channel_tilt))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_outputs(tokens, timesteps=timesteps, t_embed=t_embed, channel_tilt=channel_tilt).logits


class RGB8ContinuousDDPMModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 256,
        image_size: int = 32,
        embed_dim: int = 64,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.1,
        time_embed_input_dim: int = 128,
        time_embed_dim: int = 512,
        input_skip: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.image_size = int(image_size)
        self.n_token_channels = 3
        self.max_seq_len = self.image_size * self.image_size * self.n_token_channels
        self.expects_per_image_snr = True
        del embed_dim
        self.embed_dim = 1
        self.x_values = nn.Embedding(self.vocab_size, self.embed_dim)
        with torch.no_grad():
            values = torch.linspace(-32.0, 32.0, self.vocab_size, dtype=torch.float32).unsqueeze(1)
            self.x_values.weight.copy_(values)
        self.x_values.weight.requires_grad_(False)
        self.backbone = CIFAR10DDPMUNet(
            image_size=self.image_size,
            in_channels=3,
            out_channels=3 * self.vocab_size,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            dropout=dropout,
            time_embed_input_dim=time_embed_input_dim,
            time_embed_dim=time_embed_dim,
        )
        self.logits_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.input_skip = bool(input_skip)
        if self.input_skip:
            self.input_skip_logits = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
            self.input_skip_gate = nn.Linear(int(time_embed_dim), self.vocab_size)
            nn.init.zeros_(self.input_skip_logits.weight)
            nn.init.zeros_(self.input_skip_gate.weight)
            nn.init.zeros_(self.input_skip_gate.bias)

    def _gaussian_channel_tilt(self, noisy_embeddings: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        snr_t = torch.clamp(snr, min=1.0e-12).to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        x_values = self.x_values.weight.to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)  # [V, D]
        distances = torch.sum((noisy_embeddings.unsqueeze(2) - x_values.view(1, 1, self.vocab_size, self.embed_dim)) ** 2, dim=-1)
        return 0.5 * float(self.embed_dim) * torch.log(snr_t) - 0.5 * snr_t * distances

    def _snr_normalized_pixels(self, noisy_embeddings: torch.Tensor, snr_image: torch.Tensor) -> torch.Tensor:
        snr_scale = 1.0 + torch.sqrt(torch.clamp(snr_image, min=1.0e-12)).view(-1, 1, 1).to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        return noisy_embeddings / snr_scale

    def _input_skip_logits(self, noisy_embeddings: torch.Tensor, timesteps: torch.Tensor, snr_image: torch.Tensor) -> torch.Tensor:
        if not self.input_skip:
            return torch.zeros((*noisy_embeddings.shape[:2], self.vocab_size), dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        skip_logits = self.input_skip_logits(self._snr_normalized_pixels(noisy_embeddings, snr_image))
        time_gate = 1.0 + self.input_skip_gate(self.backbone.time_embed(timesteps)).unsqueeze(1)
        return skip_logits * time_gate.to(dtype=skip_logits.dtype)

    def forward_outputs(
        self,
        noisy_embeddings: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        del t_embed
        if noisy_embeddings.ndim != 3 or int(noisy_embeddings.shape[1]) != self.max_seq_len or int(noisy_embeddings.shape[2]) != self.embed_dim:
            raise ValueError(f"Expected noisy_embeddings [B, {self.max_seq_len}, {self.embed_dim}], got {tuple(noisy_embeddings.shape)}")
        if snr.shape != (*noisy_embeddings.shape[:2], 1):
            raise ValueError(f"Expected snr [B, {self.max_seq_len}, 1], got {tuple(snr.shape)}")
        batch_size = int(noisy_embeddings.shape[0])
        snr_t = torch.clamp(torch.as_tensor(snr, dtype=noisy_embeddings.dtype, device=noisy_embeddings.device), min=1.0e-12)
        snr_per_image = snr_t[:, :1, :]
        if not bool(torch.equal(snr_t, snr_per_image.expand_as(snr_t))):
            raise ValueError("RGB8ContinuousDDPMModel expects one SNR per image, shared by all image pixels/channels.")
        snr_image = snr_t[:, 0, 0]
        noisy_px = self._snr_normalized_pixels(noisy_embeddings, snr_image)
        noisy_image = noisy_px.view(batch_size, self.image_size, self.image_size, 3).permute(0, 3, 1, 2).contiguous()
        logits_image = self.backbone(noisy_image, torch.log(snr_image))
        logits_ext = _normalized_image_to_rgb8_logits(logits_image) + self.logits_bias.view(1, 1, self.vocab_size).to(dtype=logits_image.dtype, device=logits_image.device)
        if self.input_skip:
            logits_ext = logits_ext + self._input_skip_logits(noisy_embeddings, torch.log(snr_image), snr_image).to(dtype=logits_ext.dtype)
        tilt = self._gaussian_channel_tilt(noisy_embeddings, snr_t) if channel_tilt is None else channel_tilt
        logits = apply_channel_tilt(logits_ext, tilt)
        emb_hat = posterior_mean_from_logits(logits, self.x_values.weight.to(dtype=logits.dtype, device=logits.device))
        return TransformerModelOutput(hidden=logits_ext, logits_ext=logits_ext, logits=logits, emb_hat=emb_hat)

    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward_outputs(noisy_embeddings, snr, t_embed=t_embed, channel_tilt=channel_tilt)
        return outputs.emb_hat, outputs.logits_ext, outputs.logits

    def forward_from_tokens(
        self,
        tokens: torch.Tensor,
        snr: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        clean = self.x_values(tokens.long())
        noisy = awgn(clean, snr)
        emb_hat, logits_ext, logits = self(noisy, snr)
        return {
            "clean": clean,
            "noisy": noisy,
            "emb_hat": emb_hat,
            "logits_ext": logits_ext,
            "logits": logits,
        }

    @staticmethod
    def dequantize_mean_to_rgb_tokens(values: torch.Tensor) -> torch.Tensor:
        return rgb8_normalized_values_to_tokens(values)


class RGB8DistributionDDPMModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 256,
        image_size: int = 32,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.1,
        time_embed_input_dim: int = 128,
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.image_size = int(image_size)
        self.n_token_channels = 3
        self.max_seq_len = self.image_size * self.image_size * self.n_token_channels
        self.expects_per_image_snr = True
        self.x_values = nn.Embedding(self.vocab_size, 1)
        with torch.no_grad():
            self.x_values.weight.copy_(
                (torch.arange(self.vocab_size, dtype=torch.float32) - 0.5 * float(self.vocab_size - 1)).unsqueeze(1)
            )
        self.backbone = CIFAR10DDPMUNet(
            image_size=self.image_size,
            in_channels=3,
            out_channels=3 * self.vocab_size,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            dropout=dropout,
            time_embed_input_dim=time_embed_input_dim,
            time_embed_dim=time_embed_dim,
        )

    def _gaussian_channel_tilt(self, noisy_embeddings: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        snr_t = torch.clamp(snr, min=1.0e-12).to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        x_values = self.x_values.weight.to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        distances = (noisy_embeddings.unsqueeze(2) - x_values.view(1, 1, self.vocab_size, 1)) ** 2
        return 0.5 * torch.log(snr_t) - 0.5 * snr_t * distances.squeeze(-1)

    def _posterior_mean(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        x_values = self.x_values.weight.squeeze(-1).to(dtype=probs.dtype, device=probs.device)
        return torch.sum(probs * x_values.view(1, 1, self.vocab_size), dim=-1, keepdim=True)

    def forward_outputs(
        self,
        noisy_embeddings: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        del t_embed
        if noisy_embeddings.ndim != 3 or int(noisy_embeddings.shape[1]) != self.max_seq_len or int(noisy_embeddings.shape[2]) != 1:
            raise ValueError(f"Expected noisy_embeddings [B, {self.max_seq_len}, 1], got {tuple(noisy_embeddings.shape)}")
        if snr.shape != (*noisy_embeddings.shape[:2], 1):
            raise ValueError(f"Expected snr [B, {self.max_seq_len}, 1], got {tuple(snr.shape)}")
        batch_size = int(noisy_embeddings.shape[0])
        snr_t = torch.clamp(torch.as_tensor(snr, dtype=noisy_embeddings.dtype, device=noisy_embeddings.device), min=1.0e-12)
        snr_per_image = snr_t[:, :1, :]
        if not bool(torch.equal(snr_t, snr_per_image.expand_as(snr_t))):
            raise ValueError("RGB8DistributionDDPMModel expects one SNR per image, shared by all image pixels/channels.")
        snr_image = snr_t[:, 0, 0]
        noisy_image = noisy_embeddings.view(batch_size, self.image_size, self.image_size, 3).permute(0, 3, 1, 2).contiguous()
        logits_image = self.backbone(noisy_image, torch.log(snr_image))
        logits_ext = _normalized_image_to_rgb8_logits(logits_image)
        emb_hat = self._posterior_mean(logits_ext)
        tilt = self._gaussian_channel_tilt(noisy_embeddings, snr_t) if channel_tilt is None else channel_tilt
        logits = apply_channel_tilt(logits_ext, tilt)
        return TransformerModelOutput(hidden=logits_ext, logits_ext=logits_ext, logits=logits, emb_hat=emb_hat)

    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward_outputs(noisy_embeddings, snr, t_embed=t_embed, channel_tilt=channel_tilt)
        return outputs.emb_hat, outputs.logits_ext, outputs.logits
