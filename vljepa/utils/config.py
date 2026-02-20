from __future__ import annotations

"""Configuration helpers for reproducible VL-JEPA experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TrainConfig:
    """Typed subset of train config keys commonly consumed by entrypoints."""

    seed: int
    device: str
    precision: str
    output_dir: str
    gradient_accumulation_steps: int
    clip_grad_norm: float
    log_every: int
    save_every: int
    max_steps: int


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML into a plain dictionary."""

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must parse to a dictionary: {path}")
    return data


def validate_train_config(
    cfg: Dict[str, Any],
    *,
    require_runtime: bool = False,
    require_train: bool = True,
) -> None:
    """
    Validate required config structure and key fields.

    This keeps experiment setup explicit and fails early with actionable errors.
    """

    required = ["model", "data"]
    if require_train:
        required.append("train")
    if require_runtime:
        required.append("runtime")
    for section in required:
        if section not in cfg:
            raise KeyError(f"Missing required config section: `{section}`")

    if require_train:
        train_required = {
            "seed",
            "device",
            "precision",
            "output_dir",
            "batch_size",
            "max_steps",
            "learning_rate",
        }
        missing_train = sorted(train_required - set(cfg["train"].keys()))
        if missing_train:
            raise KeyError(f"Missing required `train` keys: {missing_train}")

        if cfg["train"]["max_steps"] <= 0:
            raise ValueError("`train.max_steps` must be > 0")
        if cfg["train"]["batch_size"] <= 0:
            raise ValueError("`train.batch_size` must be > 0")

    manifests = cfg["data"].get("train_manifests")
    if manifests is not None:
        if not isinstance(manifests, list) or not manifests:
            raise ValueError("`data.train_manifests` must be a non-empty list")


def dump_resolved_config(cfg: Dict[str, Any], output_dir: str | Path) -> Path:
    """
    Persist the resolved config used for a run for strict reproducibility.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolved_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
