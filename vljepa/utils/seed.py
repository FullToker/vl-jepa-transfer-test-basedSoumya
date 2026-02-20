"""Randomness control utilities for reproducible training/inference."""

import os
import random

import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """
    Seed Python and PyTorch RNGs.

    deterministic=True enables deterministic algorithm paths where possible.
    Note that exact bitwise reproducibility can still depend on hardware/driver.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
