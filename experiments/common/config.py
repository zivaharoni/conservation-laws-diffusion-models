"""Hydra config composition helpers for experiment entrypoints."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from .experiment_schema import apply_root_schema, register_experiment_schemas


def resolve_config_name_and_overrides(argv: list[str]) -> tuple[str, list[str]]:
    config_name = "experiments/markov_scaling/immse"
    overrides: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--config-name":
            if idx + 1 >= len(argv):
                raise ValueError("--config-name requires a value")
            config_name = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("config_name="):
            config_name = arg.split("=", 1)[1]
            idx += 1
            continue
        overrides.append(arg)
        idx += 1
    return config_name, overrides


def compose_config(project_root: str | Path, argv: list[str]) -> DictConfig:
    register_experiment_schemas()
    config_name, overrides = resolve_config_name_and_overrides(argv)
    config_dir = Path(project_root) / "configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return apply_root_schema(cfg)
