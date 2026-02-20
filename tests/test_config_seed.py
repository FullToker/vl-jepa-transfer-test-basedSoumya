"""Unit tests for config validation and deterministic seed behavior."""

import random

import torch

from vljepa.utils.config import validate_train_config
from vljepa.utils.seed import set_seed


def _minimal_cfg() -> dict:
    return {
        "model": {"vision_backbone": "vit_b_16"},
        "data": {"train_manifests": ["dummy.jsonl"]},
        "train": {
            "seed": 42,
            "device": "cpu",
            "precision": "bf16",
            "output_dir": "outputs/tmp",
            "batch_size": 2,
            "max_steps": 2,
            "learning_rate": 1e-4,
        },
    }


def test_validate_train_config_accepts_minimal_valid_structure() -> None:
    cfg = _minimal_cfg()
    validate_train_config(cfg)


def test_set_seed_is_deterministic_for_torch_and_python() -> None:
    set_seed(123, deterministic=True)
    a_py = random.random()
    a_torch = torch.rand(3)
    set_seed(123, deterministic=True)
    b_py = random.random()
    b_torch = torch.rand(3)
    assert a_py == b_py
    assert torch.equal(a_torch, b_torch)

