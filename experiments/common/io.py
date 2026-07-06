"""Output helpers for configs and metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_config(path: str | Path, cfg) -> None:
    Path(path).write_text(OmegaConf.to_yaml(cfg, resolve=True))


def deterministic_output_dir(project_root: str | Path, results_root: str, experiment_name: str, run_id: str) -> Path:
    return ensure_dir(Path(project_root) / results_root / experiment_name / run_id)
