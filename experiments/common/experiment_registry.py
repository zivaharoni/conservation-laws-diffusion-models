"""Experiment-kind registry metadata."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from .experiment_schema import (
    CausalCeExperimentConfig,
    ExitExperimentConfig,
    ImmseExperimentConfig,
    ImgExitExperimentConfig,
    ImgImmseExperimentConfig,
    ImgQscExperimentConfig,
    QscExperimentConfig,
)


@dataclass(frozen=True)
class ExperimentRuntimeSpec:
    kind: str
    experiment_type: type
    validation_name: str
    evaluation_name: str
    callback_name: str
    history_format: str
    artifact_format: str


_REGISTRY: dict[str, ExperimentRuntimeSpec] = {
    "immse": ExperimentRuntimeSpec(
        kind="immse",
        experiment_type=ImmseExperimentConfig,
        validation_name="experiments.common.runtime.validate_config",
        evaluation_name="experiments.common.evaluation.evaluate_model",
        callback_name="experiments.common.training_callbacks.build_training_callback",
        history_format="immse",
        artifact_format="immse",
    ),
    "exit": ExperimentRuntimeSpec(
        kind="exit",
        experiment_type=ExitExperimentConfig,
        validation_name="scripts.train_exit._validate_config",
        evaluation_name="scripts.train_exit._run_exit_evaluation",
        callback_name="scripts.train_exit.{_maybe_apply_lr_decay,_maybe_log_min_embed_distance,_maybe_plot_exit_curve,_maybe_early_stop}",
        history_format="exit",
        artifact_format="exit",
    ),
    "qsc": ExperimentRuntimeSpec(
        kind="qsc",
        experiment_type=QscExperimentConfig,
        validation_name="scripts.train_bsc._validate_config",
        evaluation_name="scripts.train_bsc._run_bsc_eval",
        callback_name="scripts.train_bsc.run_experiment",
        history_format="qsc",
        artifact_format="qsc",
    ),
    "causal_ce": ExperimentRuntimeSpec(
        kind="causal_ce",
        experiment_type=CausalCeExperimentConfig,
        validation_name="scripts.train_causal_ce.main",
        evaluation_name="scripts.train_causal_ce._evaluate_cross_entropy",
        callback_name="scripts.train_causal_ce._train_family",
        history_format="causal_ce",
        artifact_format="causal_ce",
    ),
    "img_exit": ExperimentRuntimeSpec(
        kind="img_exit",
        experiment_type=ImgExitExperimentConfig,
        validation_name="scripts.train_img_exit._validate_cfg",
        evaluation_name="src.lm_objectives.evaluate_exit_curve",
        callback_name="scripts.train_img_exit.main",
        history_format="img_exit",
        artifact_format="img_exit",
    ),
    "img_immse": ExperimentRuntimeSpec(
        kind="img_immse",
        experiment_type=ImgImmseExperimentConfig,
        validation_name="scripts.train_img_immse._validate_cfg",
        evaluation_name="src.lm_objectives.evaluate_immse_curve",
        callback_name="scripts.train_img_immse.main",
        history_format="img_immse",
        artifact_format="img_immse",
    ),
    "img_qsc": ExperimentRuntimeSpec(
        kind="img_qsc",
        experiment_type=ImgQscExperimentConfig,
        validation_name="scripts.train_img_qsc._validate_cfg",
        evaluation_name="src.lm_objectives.evaluate_qsc_curve",
        callback_name="scripts.train_img_qsc.main",
        history_format="img_qsc",
        artifact_format="img_qsc",
    ),
}


def get_experiment_runtime(cfg: DictConfig) -> ExperimentRuntimeSpec:
    kind = str(cfg.kind).lower()
    if kind not in _REGISTRY:
        raise ValueError(f"Unsupported kind={kind!r}.")
    return _REGISTRY[kind]


def require_experiment_kind(cfg: DictConfig, expected_kind: str) -> ExperimentRuntimeSpec:
    runtime = get_experiment_runtime(cfg)
    if runtime.kind != expected_kind:
        raise ValueError(
            f"Expected kind={expected_kind!r}, got {runtime.kind!r} "
            f"({runtime.experiment_type.__name__})."
        )
    return runtime
