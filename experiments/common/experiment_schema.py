"""Structured Hydra schemas for runnable experiment configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf


@dataclass(slots=True)
class LrDecayConfig:
    enabled: bool = True
    check_every: int = 5000
    warmup_steps: int = 1000
    min_lr: float = 1.0e-6


@dataclass(slots=True)
class PrintMinEmbedDistanceConfig:
    enabled: bool = True
    check_every: int = 5000
    large_alphabet_threshold: int = 4096


@dataclass(slots=True)
class EarlyStoppingConfig:
    enabled: bool = True
    min_steps: int = 1000
    check_every: int = 5000
    rel_error_threshold: float = 0.01
    n_snr_points: int | None = 24
    n_samples: int | None = 2048
    batch_size: int | None = 32
    seq_len: int | None = None


@dataclass(slots=True)
class PlotImmseCurveConfig:
    enabled: bool = True
    check_every: int = 5000
    dir_name: str = "train_immse_curves"
    filename_template: str = "mmse_curves_step_{step:07d}.png"
    n_snr_points: int | None = None
    n_samples: int | None = None
    batch_size: int | None = 32
    seq_len: int | None = None


@dataclass(slots=True)
class PlotExitCurveConfig:
    enabled: bool = False
    check_every: int = 5000
    dir_name: str = "train_exit_curves"
    filename_template: str = "exit_curve_step_{step:07d}.png"


@dataclass(slots=True)
class ImmseCallbacksConfig:
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    lr_decay: LrDecayConfig = field(default_factory=LrDecayConfig)
    plot_immse_curve: PlotImmseCurveConfig = field(default_factory=PlotImmseCurveConfig)
    print_min_embed_distance: PrintMinEmbedDistanceConfig = field(default_factory=PrintMinEmbedDistanceConfig)


@dataclass(slots=True)
class ExitCallbacksConfig:
    early_stopping: EarlyStoppingConfig = field(
        default_factory=lambda: EarlyStoppingConfig(enabled=False, n_snr_points=None, n_samples=None, batch_size=None)
    )
    lr_decay: LrDecayConfig = field(default_factory=LrDecayConfig)
    plot_exit_curve: PlotExitCurveConfig = field(default_factory=PlotExitCurveConfig)
    print_min_embed_distance: PrintMinEmbedDistanceConfig = field(default_factory=PrintMinEmbedDistanceConfig)


@dataclass(slots=True)
class ImmseLossConfig:
    name: str = "nll"
    label_smoothing: float = 0.0
    estimator: str = "regression"
    topk: int | None = None
    weight_loss_by_snr: bool = True


@dataclass(slots=True)
class ImmseEvalConfig:
    n_snr_points: int = 24
    n_samples: int = 4096
    batch_size: int = 512
    seq_len: int = 64
    steps_per_batch: int = 1


@dataclass(slots=True)
class ExitEvalConfig:
    eval_batches: int = 8
    eval_batch_size: int = 64
    n_s_points: int = 33
    eval_every: int | None = None
    steps_per_batch: int = 1


@dataclass(slots=True)
class CausalCeEvalConfig:
    eval_batches: int = 8
    seq_len: int = 128


@dataclass(slots=True)
class MaskingConfig:
    train_mask_min: float = 0.0
    train_mask_max: float = 1.0


@dataclass(slots=True)
class QscConfig:
    train_t_min: float = 0.00390625
    train_t_max: float = 0.9999


@dataclass(slots=True)
class BaseExperimentConfig:
    kind: str = MISSING


@dataclass(slots=True)
class ImmseExperimentConfig(BaseExperimentConfig):
    kind: str = "immse"
    loss: ImmseLossConfig = field(default_factory=ImmseLossConfig)
    eval: ImmseEvalConfig = field(default_factory=ImmseEvalConfig)
    callbacks: ImmseCallbacksConfig = field(default_factory=ImmseCallbacksConfig)


@dataclass(slots=True)
class ExitExperimentConfig(BaseExperimentConfig):
    kind: str = "exit"
    family: str = "markov"
    eval: ExitEvalConfig = field(default_factory=ExitEvalConfig)
    callbacks: ExitCallbacksConfig = field(default_factory=ExitCallbacksConfig)


@dataclass(slots=True)
class QscExperimentConfig(BaseExperimentConfig):
    kind: str = "qsc"
    family: str = "markov"
    eval: ExitEvalConfig = field(default_factory=ExitEvalConfig)
    callbacks: ExitCallbacksConfig = field(default_factory=ExitCallbacksConfig)
    qsc: QscConfig = field(default_factory=QscConfig)


@dataclass(slots=True)
class CausalCeExperimentConfig(BaseExperimentConfig):
    kind: str = "causal_ce"
    family: str = "markov"
    eval: CausalCeEvalConfig = field(default_factory=CausalCeEvalConfig)


@dataclass(slots=True)
class ImgExitExperimentConfig(BaseExperimentConfig):
    kind: str = "img_exit"
    eval: ExitEvalConfig = field(default_factory=ExitEvalConfig)


@dataclass(slots=True)
class ImgImmseExperimentConfig(BaseExperimentConfig):
    kind: str = "img_immse"
    loss: ImmseLossConfig = field(default_factory=ImmseLossConfig)
    eval: ImmseEvalConfig = field(default_factory=ImmseEvalConfig)


@dataclass(slots=True)
class ImgQscEvalConfig:
    primary_split: str = "val"
    report_splits: list[str] = field(default_factory=lambda: ["val"])
    batch_size: int = 32
    eval_every: int | None = 10000
    max_batches: int | None = 16
    n_t_points: int = 33
    t_min: float = 0.00390625
    t_max: float = 0.9999
    log_conservation_integral: bool = True
    generate_curves: bool = True
    steps_per_batch: int = 1


@dataclass(slots=True)
class ImgQscExperimentConfig(BaseExperimentConfig):
    kind: str = "img_qsc"
    eval: ImgQscEvalConfig = field(default_factory=ImgQscEvalConfig)


@dataclass(slots=True)
class RootTrainConfig:
    seed: int = 2026
    device: str = "auto"
    deterministic: bool = False
    verbose: bool = False
    name: str = MISSING
    data: Any = MISSING
    model: Any = MISSING
    train: Any = MISSING
    logging: Any = MISSING
    paths: Any = MISSING
    kind: str = MISSING
    family: str | None = None
    objective: str | None = None
    eval: Any = None
    callbacks: Any = None
    loss: Any = None
    masking: Any = None
    qsc: Any = None
    snr_min: float | None = None
    snr_max: float | None = None


_SCHEMAS_REGISTERED = False


def register_experiment_schemas() -> None:
    global _SCHEMAS_REGISTERED
    if _SCHEMAS_REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(group="root_schema", name="base", node=RootTrainConfig)
    _SCHEMAS_REGISTERED = True


def apply_root_schema(cfg: DictConfig) -> DictConfig:
    register_experiment_schemas()
    experiment_kind = str(cfg.kind).lower()
    if experiment_kind == "immse":
        schema = RootTrainConfig(
            kind="immse",
            objective="immse",
            eval=ImmseEvalConfig(),
            callbacks=ImmseCallbacksConfig(),
            loss=ImmseLossConfig(),
        )
    elif experiment_kind == "exit":
        schema = RootTrainConfig(
            kind="exit",
            family="markov",
            objective="exit",
            eval=ExitEvalConfig(),
            callbacks=ExitCallbacksConfig(),
            masking=MaskingConfig(),
        )
    elif experiment_kind == "qsc":
        schema = RootTrainConfig(
            kind="qsc",
            family="markov",
            objective="qsc",
            eval=ExitEvalConfig(),
            callbacks=ExitCallbacksConfig(),
            qsc=QscConfig(),
        )
    elif experiment_kind == "causal_ce":
        schema = RootTrainConfig(
            kind="causal_ce",
            family="markov",
            objective="causal_ce",
            eval=CausalCeEvalConfig(),
        )
    elif experiment_kind == "img_exit":
        schema = RootTrainConfig(
            kind="img_exit",
            objective="exit",
            eval=ImgQscEvalConfig(),
            masking=MaskingConfig(),
        )
    elif experiment_kind == "img_immse":
        schema = RootTrainConfig(
            kind="img_immse",
            objective="immse",
            eval=ImmseEvalConfig(),
            loss=ImmseLossConfig(),
        )
    elif experiment_kind == "img_qsc":
        schema = RootTrainConfig(
            kind="img_qsc",
            objective="qsc",
            eval=ImgQscEvalConfig(),
            qsc=QscConfig(),
        )
    else:
        raise ValueError(f"Unsupported kind={cfg.kind!r}.")

    merged = OmegaConf.merge(OmegaConf.structured(schema), cfg)
    OmegaConf.set_struct(merged, True)
    return merged
