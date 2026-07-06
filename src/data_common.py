"""Shared data runtime types."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TrainingBatch:
    states: torch.Tensor
    snr: torch.Tensor


@dataclass(slots=True)
class EvaluationParams:
    n_snr_points: int
    n_samples: int
    batch_size: int
    seq_len: int
